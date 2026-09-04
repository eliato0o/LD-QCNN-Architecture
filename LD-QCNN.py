import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import math
import numpy as np
import matplotlib.pyplot as plt
import time

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np




# ============================================================================
# 1. WARMSTART TN
# ============================================================================







# ============================================================================
# 2. PARAMÈTRES GLOBAUX
# ============================================================================
NUM_QUBITS = 16
BATCH_SIZE = 16  # Réduit pour éviter la saturation VRAM
EPOCHS = 5
TARGET_K = 4   # On change entre 4 et 2 pour voir les différences
DEPTH = 2  # Pour être sûr d'avoir une expressivité maximale

# Configuration du simulateur haute performance avec méthode adjointe
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

# ============================================================================
# 3. PRÉPARATION DE LA BASE DE DONNÉES DE RÉFÉRENCE (RSNA PNEUMONIA)
# ============================================================================
def load_and_prepare_data(data_dir='./rsna_pneumonia_data'):
    print("Chargement du dataset de référence (RSNA Pneumonia Detection)...")
    
    # Prétraitement strict exigé par le protocole : niveaux de gris et redimensionnement
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Chargement classique d'images réparties dans des dossiers par classe
    # L'arborescence attendue est data_dir/Classe_0/ et data_dir/Classe_1/
    full_dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Séparation en sous-ensembles d'entraînement (80%) et de test (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # BATCH_SIZE doit correspondre à la variable globale définie au début du script (ex: 16)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"Images d'entraînement : {len(train_dataset)}")
    print(f"Images de test : {len(test_dataset)}")
    
    return train_loader, test_loader

# ============================================================================
# 4. SOUS-BLOCS QUANTIQUES ET RÉDUCTION (LD-QCNN)
# ============================================================================
def dense_subblock(params_tensor, wires, depth):
    idx = 0
    num_wires = len(wires)
    if num_wires == 1:
        for _ in range(depth):
            qml.RX(params_tensor[idx], wires=wires[0]); idx += 1
            qml.RY(params_tensor[idx], wires=wires[0]); idx += 1
            qml.RZ(params_tensor[idx], wires=wires[0]); idx += 1
        return

    for _ in range(depth):
        for w in wires:
            qml.RX(params_tensor[idx], wires=w); idx += 1
            qml.RY(params_tensor[idx], wires=w); idx += 1
        for i in range(num_wires - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
            qml.RZ(params_tensor[idx], wires=wires[i+1]); idx += 1
        qml.CNOT(wires=[wires[-1], wires[0]])
        qml.RZ(params_tensor[idx], wires=wires[0]); idx += 1
        for w in wires:
            qml.RY(params_tensor[idx], wires=w); idx += 1

def pooling_layer(phi, wire_source, wire_target):
    qml.CNOT(wires=[wire_source, wire_target])
    qml.RY(phi, wires=wire_target)
    qml.CNOT(wires=[wire_source, wire_target])

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def ld_qcnn_circuit(inputs, conv_params, pool_params, active_wires_architecture, k_val, depth):
    # Encodage d'amplitude : nécessite un vecteur normalisé de taille 2^16 = 65536
    qml.AmplitudeEmbedding(features=inputs, wires=range(NUM_QUBITS), normalize=True)
    
    stage_idx = 0
    active_wires = list(range(NUM_QUBITS))
    
    while len(active_wires) > 2:
        num_blocks = math.ceil(len(active_wires) / k_val)
        for b in range(num_blocks):
            wires_block = active_wires[b*k_val : min((b+1)*k_val, len(active_wires))]
            if len(wires_block) >= 1:
                needed_params = depth * 4 * len(wires_block) if len(wires_block) > 1 else depth * 3
                block_params = conv_params[stage_idx][b][:needed_params]
                dense_subblock(block_params, wires_block, depth)
                
        next_wires = []
        pool_idx = 0
        for i in range(0, len(active_wires)-1, 2):
            pooling_layer(pool_params[stage_idx][pool_idx], active_wires[i], active_wires[i+1])
            next_wires.append(active_wires[i+1])
            pool_idx += 1
            
        if len(active_wires) % 2 != 0:
             next_wires.append(active_wires[-1])
             
        active_wires = next_wires
        stage_idx += 1
        
    if len(active_wires) > 0:
        needed_params = depth * 4 * len(active_wires)
        dense_subblock(conv_params[stage_idx][0][:needed_params], active_wires, depth)
    
    # Mesure de PauliZ sur les qubits restants
    return [qml.expval(qml.PauliZ(w)) for w in active_wires]

# ============================================================================
# 5. MODÈLE PYTORCH HYBRIDE
# ============================================================================
class LDQCNNModule(nn.Module):
    def __init__(self, k=4, depth=2):
        super(LDQCNNModule, self).__init__()
        self.k = k
        self.depth = depth
        self.active_wires_architecture = []
        
        active_wires = NUM_QUBITS
        self.conv_params = nn.ParameterList()
        self.pool_params = nn.ParameterList()
        
        while active_wires > 2:
            num_blocks = math.ceil(active_wires / self.k)
            max_params_per_block = self.depth * 4 * self.k
            self.conv_params.append(nn.Parameter(torch.randn(num_blocks, max_params_per_block) * 0.1))
            
            num_pools = active_wires // 2
            self.pool_params.append(nn.Parameter(torch.randn(num_pools) * 0.1))
            
            self.active_wires_architecture.append(active_wires)
            active_wires = active_wires // 2 + (active_wires % 2)
            
        self.active_wires_architecture.append(active_wires)
        if active_wires > 0:
            final_params = self.depth * 4 * active_wires
            self.conv_params.append(nn.Parameter(torch.randn(1, final_params) * 0.1))


    # Pour attribuer les poids trouvés pendant le warmstart
    def load_warmstart_weights(self, theta_seed, device='cpu'):
        """
        Injecte les poids optimisés du warm-start dans les paramètres du modèle.
        theta_seed est une liste de listes (structure retournée par votre fonction de warm-start)
        """
        print("Initialisation des poids du LD-QCNN avec theta_seed...")
        
        # Supposons que vous avez une liste de vos paramètres de couches dans self.layers
        # Adaptez 'self.layers' selon votre nom de variable réel
        for l_idx, layer_params in enumerate(theta_seed):
            for b_idx, weight_tensor in enumerate(layer_params):
                
                # Accès au paramètre correspondant dans votre modèle
                # Ici, on suppose une structure hiérarchique identique
                target_param = self.layers[l_idx][b_idx] 
                
                # Copie des données
                with torch.no_grad():
                    target_param.data = weight_tensor.clone().to(device)
                    target_param.requires_grad = True # Très important pour le fine-tuning
        
        print("Weights successfully loaded.")

    def forward(self, x):
        # Aplatissement [Batch, 256, 256] -> [Batch, 65536]
        x_flat = x.view(x.shape[0], -1).to(torch.float64)
        
        # Normalisation L2 obligatoire pour l'AmplitudeEmbedding
        x_norm = torch.nn.functional.normalize(x_flat, p=2, dim=1)
        
        predictions = []
        for sample in x_norm:
            raw_out = ld_qcnn_circuit(
                sample, self.conv_params, self.pool_params, 
                self.active_wires_architecture, self.k, self.depth
            )
            # Transformation de l'observable [-1, 1] en probabilité [0, 1]
            out_tensor = torch.stack(raw_out)
            prob = 0.5 * (1.0 - out_tensor.mean())
            predictions.append(prob)
            
        return torch.stack(predictions)

# ============================================================================
# 6. BOUCLE D'ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================
def calculate_accuracy(y_pred, y_true):
    preds = (y_pred >= 0.5).float()
    correct = (preds == y_true).float().sum()
    return correct / len(y_true)

def main():
    train_loader, test_loader = load_and_prepare_data()
    
    model = LDQCNNModule(k=TARGET_K, depth=DEPTH)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n--- Démarrage de l'entraînement (k={TARGET_K}, qubits={NUM_QUBITS}) ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        batch_accs = []
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.to(outputs.dtype))
            
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            batch_accs.append(calculate_accuracy(outputs, labels).item())
            
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_accs)
        
        # Phase de validation
        model.eval()
        val_losses = []
        val_accs = []
        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_outputs = model(val_images)
                v_loss = criterion(val_outputs, val_labels.to(val_outputs.dtype))
                val_losses.append(v_loss.item())
                val_accs.append(calculate_accuracy(val_outputs, val_labels).item())
                
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accs)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    print(f"Temps total d'exécution : {time.time() - start_time:.2f} secondes")

    # ============================================================================
    # 7. VISUALISATION DES RÉSULTATS (MATPLOTLIB)
    # ============================================================================
    epochs_arr = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Graphique de la Précision
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, history['train_acc'], 'g', label='Training acc')
    plt.plot(epochs_arr, history['val_acc'], 'black', label='Validation acc')
    plt.title('Training and Validation Accuracy (LD-QCNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Graphique de la Perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, history['train_loss'], 'g', label='Training loss')
    plt.plot(epochs_arr, history['val_loss'], 'black', label='Validation loss')
    plt.title('Training and Validation Loss (LD-QCNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



    # ============================================================================
    # 8. VISUALISATION DU CIRCUIT
    # ============================================================================


def afficher_diagramme_circuit():
    print("Génération du diagramme du circuit (cela peut prendre quelques secondes)...")
    
    # Paramètres de l'architecture
    k_val = 2
    depth = 2
    
    # Instanciation du modèle PyTorch pour récupérer l'architecture et les poids
    model = LDQCNNModule(k=k_val, depth=depth)
    
    # Génération d'une donnée d'entrée factice strictement normalisée (1 image)
    dummy_input = torch.rand(65536, dtype=torch.float64)
    dummy_input = torch.nn.functional.normalize(dummy_input.unsqueeze(0), p=2, dim=1)[0]
    
    # Utilisation de qml.draw_mpl pour générer le diagramme style "publication"
    # Le style "pennylane" ou "default" produit le rendu académique standard
    fig, ax = qml.draw_mpl(
        ld_qcnn_circuit, 
        style="pennylane",
        decimals=2 # N'affiche que 2 décimales pour les paramètres de rotation
    )(
        dummy_input, 
        model.conv_params, 
        model.pool_params, 
        model.active_wires_architecture, 
        k_val, 
        depth
    )
    
    plt.title("Diagramme de l'Architecture LD-QCNN (16 qubits, k=4)", fontsize=16)
    
    # Affichage de l'image (vous pourrez zoomer et naviguer dedans)
    plt.show()

















######### IL MANQUE LA CONNEXION ENTRE LES POIDS TROUVES PAR LE TN ET LE MODELE ############

if __name__ == "__main__":
    main()
    # Décommentez cette ligne pour l'exécuter directement à la fin de votre script :
    afficher_diagramme_circuit()