import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import pennylane as qml
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# 1. PARAMÈTRES GLOBAUX DU PROTOCOLE
# ============================================================================
NUM_QUBITS = 16
CHI = 16        # Dimension de liaison maximale
TARGET_K = 4    # Sous-blocs k=4 (2^4 = 16 dimensions d'états)
DEPTH = 2       # Profondeur des sous-blocs convolutifs
BATCH_SIZE = 16
EPOCHS = 5
EPOCHS_TNI = 20 # Nombre d'époques pour le Warm-Start classique

# Simulateur haute performance avec méthode adjointe
dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

# ============================================================================
# 2. WARMSTART TN (TTN CLASSIQUE & SYNTHÈSE UNITAIRE)
# ============================================================================

class ClassicalTTN(nn.Module):
    """
    Émulateur classique du LD-QCNN utilisant des contractions tensorielles denses (einsum).
    """
    def __init__(self, chi=CHI):
        super(ClassicalTTN, self).__init__()
        self.chi = chi
        
        # 1. Blocs convolutifs (matrices 16x16 agissant sur 4 qubits chacun)
        self.conv_layer1 = nn.ParameterList([nn.Parameter(torch.randn(16, 16, dtype=torch.float32) * 0.1) for _ in range(4)])
        self.conv_layer2 = nn.ParameterList([nn.Parameter(torch.randn(16, 16, dtype=torch.float32) * 0.1) for _ in range(2)])
        self.conv_layer3 = nn.ParameterList([nn.Parameter(torch.randn(16, 16, dtype=torch.float32) * 0.1) for _ in range(1)])
        
        # 2. Blocs de pooling (isométries de compression 16x16 -> 16)
        self.pool_layer1 = nn.ParameterList([nn.Parameter(torch.randn(16, 16, 16, dtype=torch.float32) * 0.1) for _ in range(2)])
        self.pool_layer2 = nn.Parameter(torch.randn(16, 16, 16, dtype=torch.float32) * 0.1)
        
        # 3. Observable locale
        self.observable = nn.Parameter(torch.randn(16, 16, dtype=torch.float32) * 0.1)

    def project_to_isometry(self):
        """Projette les matrices de convolution et tenseurs de pooling sur la variété unitaire."""
        with torch.no_grad():
            for param_list in [self.conv_layer1, self.conv_layer2, self.conv_layer3]:
                for param in param_list:
                    U, _, Vh = torch.linalg.svd(param)
                    param.copy_(U @ Vh)
                    
            for p in self.pool_layer1:
                U, _, Vh = torch.linalg.svd(p.view(16, 256), full_matrices=False)
                p.copy_((U @ Vh).view(16, 16, 16))
                
            U, _, Vh = torch.linalg.svd(self.pool_layer2.view(16, 256), full_matrices=False)
            self.pool_layer2.copy_((U @ Vh).view(16, 16, 16))

    def forward(self, x):
        B = x.shape[0]
        x_tensor = x.view(B, 16, 16, 16, 16).float()
        
        # Étage 1 : Convolution
        c1 = torch.einsum('ai,bj,ck,dl,nijkl->nabcd',
                          self.conv_layer1[0], self.conv_layer1[1],
                          self.conv_layer1[2], self.conv_layer1[3], x_tensor)
        
        # Étage 1 : Pooling
        p1 = torch.einsum('mab,kcd,nabcd->nmk', self.pool_layer1[0], self.pool_layer1[1], c1)
        
        # Étage 2 : Convolution
        c2 = torch.einsum('ai,bj,nij->nab', self.conv_layer2[0], self.conv_layer2[1], p1)
        
        # Étage 2 : Pooling
        p2 = torch.einsum('mab,nab->nm', self.pool_layer2, c2)
        
        # Étage 3 : Convolution finale
        c3 = torch.einsum('ai,ni->na', self.conv_layer3[0], p2)
        
        # Normalisation locale et évaluation de l'observable
        norm = torch.norm(c3, dim=1, keepdim=True) + 1e-8
        c3_norm = c3 / norm
        expval = torch.einsum('ni,ij,nj->n', c3_norm, self.observable, c3_norm)
        
        return torch.sigmoid(expval)


def extract_quantum_parameters_from_unitary(target_unitary, num_params=32):
    """Synthèse unitaire vers les angles quantiques theta."""
    theta = torch.randn(num_params, requires_grad=True, dtype=torch.float64)
    optimizer = optim.Adam([theta], lr=0.1)
    
    def block_circuit(params):
        idx = 0
        wires = [0, 1, 2, 3]
        for _ in range(2):
            for w in wires:
                qml.RX(params[idx], wires=w); idx += 1
                qml.RY(params[idx], wires=w); idx += 1
            for i in range(3):
                qml.CNOT(wires=[wires[i], wires[i+1]])
                qml.RZ(params[idx], wires=wires[i+1]); idx += 1
            qml.CNOT(wires=[wires[-1], wires[0]])
            qml.RZ(params[idx], wires=wires[0]); idx += 1
            for w in wires:
                qml.RY(params[idx], wires=w); idx += 1

    print("      -> Micro-optimisation du sous-bloc quantique vers la cible...")
    target_complex = target_unitary.to(torch.complex128)
    
    for step in range(50):
        optimizer.zero_grad()
        # Correction wire_order ajoutée ici :
        matrix_q = qml.matrix(block_circuit, wire_order=[0, 1, 2, 3])(theta)
        loss = torch.norm(matrix_q - target_complex, p='fro')
        loss.backward()
        optimizer.step()
        
    return theta.detach()


def perform_tni_warm_start(train_loader):
    """Exécute l'initialisation TNI complète."""
    print("\n=======================================================")
    print(" DÉMARRAGE DU WARM-START TN (Phases 1 & 2 : TTN Classique)")
    print("=======================================================")
    
    ttn_model = ClassicalTTN(chi=CHI)
    # Taux d'apprentissage réduit à 0.01 pour stabiliser l'optimisation sous contrainte unitaire
    optimizer_ttn = optim.Adam(ttn_model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    d_sub_images, d_sub_labels = next(iter(train_loader))
    d_sub_images = d_sub_images.view(d_sub_images.shape[0], -1)
    d_sub_images = torch.nn.functional.normalize(d_sub_images, p=2, dim=1)
    d_sub_labels = d_sub_labels.float()
    
    ttn_model.train()
    for epoch in range(EPOCHS_TNI):
        optimizer_ttn.zero_grad()
        ttn_model.project_to_isometry()
        preds = ttn_model(d_sub_images)
        loss = criterion(preds, d_sub_labels)
        loss.backward()
        
        # Écrêtage des gradients pour éviter les dérives
        torch.nn.utils.clip_grad_norm_(ttn_model.parameters(), max_norm=1.0)
        optimizer_ttn.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  [TTN] Époque {epoch+1:02d}/{EPOCHS_TNI} | Perte Classique: {loss.item():.4f}")
            
    print("\n--- PHASE 3 : SYNTHÈSE UNITAIRE VERS PARAMÈTRES QUANTIQUES ---")
    theta_seed = []
    layers = [ttn_model.conv_layer1, ttn_model.conv_layer2, ttn_model.conv_layer3]
    
    for l_idx, layer in enumerate(layers):
        print(f"Extraction & synthèse de l'étage convolutif {l_idx + 1}...")
        stage_params = []
        for block_idx, classical_tensor in enumerate(layer):
            U, _, Vh = torch.linalg.svd(classical_tensor.detach())
            U_target = U @ Vh
            q_params = extract_quantum_parameters_from_unitary(U_target, num_params=32)
            stage_params.append(q_params)
        theta_seed.append(stage_params)
        
    print(" [SUCCÈS] Vecteur theta_seed prêt pour injection.")
    return theta_seed

# ============================================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================================
def load_and_prepare_data(data_dir='./rsna_pneumonia_data'):
    print(f"\nChargement du dataset depuis '{data_dir}'...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Dataset prêt : {len(train_dataset)} train, {len(test_dataset)} test.")
    return train_loader, test_loader

# ============================================================================
# 4. SOUS-BLOCS QUANTIQUES ET CIRCUIT LD-QCNN
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
    
    return [qml.expval(qml.PauliZ(w)) for w in active_wires]

# ============================================================================
# 5. MODÈLE HYBRIDE PYTORCH
# ============================================================================
class LDQCNNModule(nn.Module):
    def __init__(self, k=TARGET_K, depth=DEPTH):
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
            self.conv_params.append(nn.Parameter(torch.randn(num_blocks, max_params_per_block, dtype=torch.float64) * 0.1))
            
            num_pools = active_wires // 2
            self.pool_params.append(nn.Parameter(torch.randn(num_pools, dtype=torch.float64) * 0.1))
            
            self.active_wires_architecture.append(active_wires)
            active_wires = active_wires // 2 + (active_wires % 2)
            
        self.active_wires_architecture.append(active_wires)
        if active_wires > 0:
            final_params = self.depth * 4 * active_wires
            self.conv_params.append(nn.Parameter(torch.randn(1, final_params, dtype=torch.float64) * 0.1))

    def load_warmstart_weights(self, theta_seed):
        """Injecte les poids optimisés theta_seed dans self.conv_params."""
        print("\n[CONNEXION] Injection des paramètres theta_seed dans LDQCNNModule...")
        with torch.no_grad():
            for l_idx, stage_weights in enumerate(theta_seed):
                if l_idx < len(self.conv_params):
                    for b_idx, block_weight in enumerate(stage_weights):
                        if b_idx < self.conv_params[l_idx].shape[0]:
                            num_w = min(len(block_weight), self.conv_params[l_idx].shape[1])
                            self.conv_params[l_idx].data[b_idx, :num_w] = block_weight[:num_w].to(
                                device=self.conv_params[l_idx].device,
                                dtype=self.conv_params[l_idx].dtype
                            )
        print("  -> Initialisation des paramètres convolutifs réussie !")

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1).to(torch.float64)
        x_norm = torch.nn.functional.normalize(x_flat, p=2, dim=1)
        
        predictions = []
        for sample in x_norm:
            raw_out = ld_qcnn_circuit(
                sample, self.conv_params, self.pool_params, 
                self.active_wires_architecture, self.k, self.depth
            )
            out_tensor = torch.stack(raw_out)
            prob = 0.5 * (1.0 - out_tensor.mean())
            predictions.append(prob)
            
        return torch.stack(predictions)

# ============================================================================
# 6. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================
def calculate_accuracy(y_pred, y_true):
    preds = (y_pred >= 0.5).float()
    correct = (preds == y_true).float().sum()
    return correct / len(y_true)

def main():
    train_loader, test_loader = load_and_prepare_data()
    
    # Exécution du Warm-Start
    theta_seed = perform_tni_warm_start(train_loader)
    
    # Création et initialisation du modèle LD-QCNN
    model = LDQCNNModule(k=TARGET_K, depth=DEPTH)
    model.load_warmstart_weights(theta_seed)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n--- Démarrage de l'entraînement LD-QCNN (k={TARGET_K}, qubits={NUM_QUBITS}) ---")
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
        
        # Validation
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

    print(f"\nTemps total d'exécution : {time.time() - start_time:.2f} secondes")

    # Visualisation des métriques
    epochs_arr = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, history['train_acc'], 'g', label='Training acc')
    plt.plot(epochs_arr, history['val_acc'], 'black', label='Validation acc')
    plt.title('Accuracy (LD-QCNN avec TNI Warm-Start)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, history['train_loss'], 'g', label='Training loss')
    plt.plot(epochs_arr, history['val_loss'], 'black', label='Validation loss')
    plt.title('Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. VISUALISATION DU CIRCUIT
# ============================================================================
def afficher_diagramme_circuit():
    print("Génération du diagramme du circuit haute résolution...")
    
    k_val = TARGET_K
    depth = DEPTH
    
    model = LDQCNNModule(k=k_val, depth=depth)
    dummy_input = torch.rand(65536, dtype=torch.float64)
    dummy_input = torch.nn.functional.normalize(dummy_input.unsqueeze(0), p=2, dim=1)[0]
    
    fig, ax = qml.draw_mpl(
        ld_qcnn_circuit, 
        style="pennylane",
        decimals=None,
        fontsize=7
    )(
        dummy_input, 
        model.conv_params, 
        model.pool_params, 
        model.active_wires_architecture, 
        k_val, 
        depth
    )
    
    fig.set_size_inches(32, 12)
    plt.title(f"Architecture LD-QCNN ({NUM_QUBITS} qubits, k={k_val}, depth={depth})", fontsize=14, pad=20)
    fig.savefig("circuit_ld_qcnn.pdf", bbox_inches='tight')
    fig.savefig("circuit_ld_qcnn.png", dpi=300, bbox_inches='tight')
    print("  -> Diagramme exporté sous 'circuit_ld_qcnn.png' et 'circuit_ld_qcnn.pdf'.")
    plt.show()

if __name__ == "__main__":
    main()
    afficher_diagramme_circuit()