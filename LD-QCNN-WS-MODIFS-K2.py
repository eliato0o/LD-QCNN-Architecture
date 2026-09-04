"""
Modifications apportées au code LD-QCNN.py pour éviter le minimum local à 78% d'accuracy.

Warm-Start multi-lots (perform_tni_warm_start) : Au lieu de s'entraîner sur un seul lot de 16 images, l'algorithme accumule 5 lots (80 images) pour représenter équitablement les classes saine et pathologique.  

Tête de classification hybride (self.head) : Ajout d'une couche linéaire 1D (nn.Linear(1, 1)) en sortie du circuit quantique pour transformer la moyenne des observables de Pauli-Z en un score (logit non borné), ce qui permet à l'optimiseur d'ajuster le biais global sans bloquer les portes quantiques.

Fonction de perte pondérée (BCEWithLogitsLoss) : Remplacement de la MSELoss par une entropie croisée binaire avec un poids pos_weight = 3.5 pour contrer le déséquilibre de classe du jeu de données RSNA et pénaliser les faux négatifs.  

Taux d'apprentissage différentiés : Configuration de l'optimiseur Adam avec 3 groupes de paramètres :conv_params à lr=0.005 (ajustement fin des poids issus du Warm-Start),pool_params à lr=0.02 (apprentissage accéléré des réductions non pré-entraînées),head à lr=0.01 (ajustement du seuil de décision).

Mise à jour du calcul d'exactitude (calculate_accuracy) : Adaptation du seuil de décision sur les logits ($\ge 0.0$, équivalent à une probabilité $\ge 0.5$).

Il y avait un problème avec le Warmstart du TN, l'observable était TOTALEMENT aléatoire ce qui biaisait directement le paramètre de l'observable du circuit quantique (il n'était absolument pas au bon endroit).

On passe à 32 images voir plus pour le batchsize, car étant donné la répartition des cas saints ou non, 16 images contiennent bien trop peu de cas de pneumonie, il doit même avoir des cas de batchsize sans cas de pneumonie.

On passe maintenant d'une learning rate de 0.005 à 0.001 (car les angles du circuit quantique sont très sensibles) et on change d'optimiseur, passe à L-BFGS au lieu de Adam.

Optimisation du temps de calcul par rapport aux caractéristiques de ma machine, en limitant notamment le nombre de threads (nombre de process) à threads = 16, ce qui correspond aux coeurs performants de mon i9 de 13ème génération.

Dans l'ancienne version, on utilisait000000 la MSELoss (qui plafonne généralement entre 0.0 et 1.0) ou une BCELoss classique (où l'hésitation totale donne environ 0.69).Maintenant, nous utilisons BCEWithLogitsLoss(pos_weight=3.5). Ce 3.5 change toute l'échelle des valeurs :
- Ton dataset contient environ 25 % de malades et 75 % de sains.
- Si le modèle hésite et prédit un logit proche de 0 (soit $\approx 50\%$ de certitude), la perte pour un patient sain est de $0.69$.
- Mais la perte pour un patient malade non détecté est multipliée par 3.5, soit $0.69 \times 3.5 \approx 2.41$ !
- En faisant la moyenne sur un batch, on obtient un calcul théorique d'environ : $(0.75 \times 0.69) + (0.25 \times 2.41) = \mathbf{1.12}$.

On a ajouté des métriques afin de voir pour chaque classe le pourcentage d'élément qui été correctement déterminer.

On change maintenant les hyperparamètres pour ce trio (Batch 64, 10 Époques, LR 0.002) afin de laisser le modèle converger.

Modifications apportées au code LD-QCNN.py pour éviter le minimum local à 78% d'accuracy.

- Ajout d'un WeightedRandomSampler pour forcer des batchs parfaitement équilibrés (50% sains / 50% malades).
- Suppression du pos_weight (devenu inutile grâce au Sampler).
- Configuration pour un entraînement stable : Batch Size de 64, 10 Époques, Learning Rate quantique de 0.002.
- Maintien de l'optimisation CPU Intel (16 threads).
- Tête de classification hybride, métriques avancées (Rappel/Spécificité) et L-BFGS pour la synthèse unitaire.


Code LD-QCNN optimisé pour k=2 (Convolutions sur 2 qubits).
- Architecture à 4 étages (16->8->4->2->1 qubits).
- TTN Classique (Warm-Start) entièrement adapté aux tenseurs 4x4.
- Synthèse unitaire adaptée pour 2 qubits (16 paramètres par bloc).
- Optimisation CPU (16 threads), Sampler équilibré, et DataLoader ajusté.

"""


import os
import multiprocessing

# ============================================================================
# 0. OPTIMISATION MATÉRIELLE POUR CPU INTEL (i9 13th Gen)
# ============================================================================
NUM_THREADS = "16"

os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_THREADS

import torch
import pennylane as qml

torch.set_num_threads(int(NUM_THREADS))
print(f"[OPTIMISATION CPU] Exécution configurée pour utiliser {NUM_THREADS} threads intensifs.")

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# 1. PARAMÈTRES GLOBAUX DU PROTOCOLE
# ============================================================================
NUM_QUBITS = 16
CHI = 4         # Dimension maximale pour k=2 (2^2 = 4)
TARGET_K = 2    # Sous-blocs k=2 (2 qubits par convolution)
DEPTH = 2       # Profondeur des portes quantiques
BATCH_SIZE = 8  # Petit batch pour maximiser les mises à jour
EPOCHS = 30
EPOCHS_TNI = 20 

dev = qml.device("lightning.qubit", wires=NUM_QUBITS)

# ============================================================================
# 2. WARMSTART TN (TTN CLASSIQUE & SYNTHÈSE UNITAIRE POUR K=2)
# ============================================================================
class ClassicalTTN(nn.Module):
    """
    Émulateur classique du LD-QCNN pour k=2 (Tenseurs de dimension 4x4).
    4 Étages de réduction : 16 -> 8 -> 4 -> 2 -> 1
    """
    def __init__(self, chi=CHI):
        super(ClassicalTTN, self).__init__()
        self.chi = chi
        
        # Convolutions
        self.conv_layer1 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, dtype=torch.float32) * 0.1) for _ in range(8)])
        self.conv_layer2 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, dtype=torch.float32) * 0.1) for _ in range(4)])
        self.conv_layer3 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, dtype=torch.float32) * 0.1) for _ in range(2)])
        self.conv_layer4 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, dtype=torch.float32) * 0.1) for _ in range(1)])
        
        # Poolings (4x4 -> 4)
        self.pool_layer1 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, 4, dtype=torch.float32) * 0.1) for _ in range(4)])
        self.pool_layer2 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, 4, dtype=torch.float32) * 0.1) for _ in range(2)])
        self.pool_layer3 = nn.ParameterList([nn.Parameter(torch.randn(4, 4, 4, dtype=torch.float32) * 0.1) for _ in range(1)])
        
        # Observable Pauli-Z moyenne sur 2 qubits
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
        I = torch.eye(2, dtype=torch.float32)
        Z_avg = (torch.kron(Z, I) + torch.kron(I, Z)) / 2.0
        self.register_buffer('observable', Z_avg)

    def forward(self, x):
        B = x.shape[0]
        # Reshape de 65536 valeurs en 8 indices de dimension 4 (4^8 = 65536)
        x_tensor = x.view(B, 4, 4, 4, 4, 4, 4, 4, 4).float()
        
        # Étage 1 (z = batch size)
        c1 = torch.einsum('ai,bj,ck,dl,em,fn,go,hp,zijklmnop->zabcdefgh',
                          self.conv_layer1[0], self.conv_layer1[1], self.conv_layer1[2], self.conv_layer1[3],
                          self.conv_layer1[4], self.conv_layer1[5], self.conv_layer1[6], self.conv_layer1[7], x_tensor)
        p1 = torch.einsum('iab,jcd,kef,lgh,zabcdefgh->zijkl',
                          self.pool_layer1[0], self.pool_layer1[1], self.pool_layer1[2], self.pool_layer1[3], c1)
        
        # Étage 2
        c2 = torch.einsum('ai,bj,ck,dl,zijkl->zabcd',
                          self.conv_layer2[0], self.conv_layer2[1], self.conv_layer2[2], self.conv_layer2[3], p1)
        p2 = torch.einsum('iab,jcd,zabcd->zij', self.pool_layer2[0], self.pool_layer2[1], c2)
        
        # Étage 3
        c3 = torch.einsum('ai,bj,zij->zab', self.conv_layer3[0], self.conv_layer3[1], p2)
        p3 = torch.einsum('iab,zab->zi', self.pool_layer3[0], c3)
        
        # Étage 4 (Final)
        c4 = torch.einsum('ai,zi->za', self.conv_layer4[0], p3)
        
        norm = torch.norm(c4, dim=1, keepdim=True) + 1e-8
        c4_norm = c4 / norm
        
        # Mesure de l'observable
        expval = torch.einsum('zi,ij,zj->z', c4_norm, self.observable, c4_norm)
        prob = 0.5 * (1.0 - expval)
        return torch.clamp(prob, min=1e-6, max=1.0-1e-6)

def extract_quantum_parameters_from_unitary(target_unitary, num_params=16):
    """Synthèse d'une matrice 4x4 vers un circuit à 2 qubits."""
    theta = torch.randn(num_params, requires_grad=True, dtype=torch.float64)
    optimizer = optim.LBFGS([theta], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")
    
    def block_circuit(params):
        idx = 0
        wires = [0, 1]
        for _ in range(DEPTH):
            for w in wires:
                qml.RX(params[idx], wires=w); idx += 1
                qml.RY(params[idx], wires=w); idx += 1
            qml.CNOT(wires=[0, 1])
            qml.RZ(params[idx], wires=1); idx += 1
            qml.CNOT(wires=[1, 0])
            qml.RZ(params[idx], wires=0); idx += 1
            for w in wires:
                qml.RY(params[idx], wires=w); idx += 1

    target_complex = target_unitary.to(torch.complex128)
    
    def closure():
        optimizer.zero_grad()
        matrix_q = qml.matrix(block_circuit, wire_order=[0, 1])(theta)
        loss = torch.norm(matrix_q - target_complex, p='fro')
        loss.backward()
        return loss

    optimizer.step(closure)
    final_loss = closure().item()
    print(f"      -> Synthèse du bloc k=2 terminée (Loss Frobenius: {final_loss:.4f})")
    return theta.detach()

def perform_tni_warm_start(train_loader):
    print("\n=======================================================")
    print(" DÉMARRAGE DU WARM-START TN (Phases 1 & 2 : TTN Classique)")
    print("=======================================================")
    
    ttn_model = ClassicalTTN(chi=CHI)
    optimizer_ttn = optim.Adam(ttn_model.parameters(), lr=0.02)
    criterion = nn.BCELoss()
    
    images_list, labels_list = [], []
    for i, (imgs, lbls) in enumerate(train_loader):
        images_list.append(imgs)
        labels_list.append(lbls)
        if i >= 4:
            break
            
    d_sub_images = torch.cat(images_list, dim=0)
    d_sub_images = d_sub_images.view(d_sub_images.shape[0], -1)
    d_sub_images = torch.nn.functional.normalize(d_sub_images, p=2, dim=1)
    d_sub_labels = torch.cat(labels_list, dim=0).float()
    
    ttn_model.train()
    for epoch in range(EPOCHS_TNI):
        optimizer_ttn.zero_grad()
        preds = ttn_model(d_sub_images)
        loss = criterion(preds, d_sub_labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(ttn_model.parameters(), max_norm=1.0)
        optimizer_ttn.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  [TTN] Époque {epoch+1:02d}/{EPOCHS_TNI} | Perte Classique: {loss.item():.4f}")
            
    print("\n--- PHASE 3 : SYNTHÈSE UNITAIRE VERS PARAMÈTRES QUANTIQUES ---")
    theta_seed = []
    layers = [ttn_model.conv_layer1, ttn_model.conv_layer2, ttn_model.conv_layer3, ttn_model.conv_layer4]
    
    for l_idx, layer in enumerate(layers):
        print(f"Extraction & synthèse de l'étage convolutif {l_idx + 1}...")
        stage_params = []
        for block_idx, classical_tensor in enumerate(layer):
            U, _, Vh = torch.linalg.svd(classical_tensor.detach())
            U_target = U @ Vh
            # num_params=16 car depth=2 et wires=2
            q_params = extract_quantum_parameters_from_unitary(U_target, num_params=16)
            stage_params.append(q_params)
        theta_seed.append(stage_params)
        
    print(" [SUCCÈS] Vecteur theta_seed prêt pour injection.")
    return theta_seed

# ============================================================================
# 3. PRÉPARATION DES DONNÉES
# ============================================================================
def load_and_prepare_data(data_dir='./brain_tumor_dataset', num_per_class=1500):
    print(f"\nChargement du dataset depuis '{data_dir}'...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(root=data_dir, transform=transform)
    
    print("Équilibrage et extraction du sous-ensemble...")
    class_0_indices = [i for i, label in enumerate(full_dataset.targets) if label == 0]
    class_1_indices = [i for i, label in enumerate(full_dataset.targets) if label == 1]
    
    n_samples = min(num_per_class, len(class_0_indices), len(class_1_indices))
    print(f" -> Extraction de {n_samples} images de classe 0 et {n_samples} images de classe 1.")
    
    random.seed(42) 
    subset_0 = random.sample(class_0_indices, n_samples)
    subset_1 = random.sample(class_1_indices, n_samples)
    
    balanced_indices = subset_0 + subset_1
    balanced_dataset = Subset(full_dataset, balanced_indices)
    
    train_size = int(0.8 * len(balanced_dataset))
    test_size = len(balanced_dataset) - train_size
    train_dataset, test_dataset = random_split(balanced_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
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
                # Calcul adaptatif du nombre de paramètres
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

        self.head = nn.Linear(1, 1, dtype=torch.float64)

    def load_warmstart_weights(self, theta_seed):
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
        
        logits = []
        for sample in x_norm:
            raw_out = ld_qcnn_circuit(
                sample, self.conv_params, self.pool_params, 
                self.active_wires_architecture, self.k, self.depth
            )
            out_tensor = torch.stack(raw_out)
            mean_expval = out_tensor.mean().unsqueeze(0).to(self.head.weight.dtype)
            logit = self.head(mean_expval)
            logits.append(logit)
            
        return torch.stack(logits).squeeze(1)

# ============================================================================
# 6. ENTRAÎNEMENT ET ÉVALUATION
# ============================================================================
def calculate_metrics(y_logits, y_true):
    preds = (y_logits >= 0.0).float()
    y_true = y_true.to(preds.device).float()
    
    TP = ((preds == 1.0) & (y_true == 1.0)).float().sum()
    TN = ((preds == 0.0) & (y_true == 0.0)).float().sum()
    FP = ((preds == 1.0) & (y_true == 0.0)).float().sum()
    FN = ((preds == 0.0) & (y_true == 1.0)).float().sum()
    
    acc = (TP + TN) / len(y_true)
    recall = TP / (TP + FN) if (TP + FN) > 0.0 else torch.tensor(float('nan'))
    specificity = TN / (TN + FP) if (TN + FP) > 0.0 else torch.tensor(float('nan'))
    
    return acc.item(), recall.item(), specificity.item()

def main():
    train_loader, test_loader = load_and_prepare_data()
    
    theta_seed = perform_tni_warm_start(train_loader)
    
    model = LDQCNNModule(k=TARGET_K, depth=DEPTH)
    model.load_warmstart_weights(theta_seed)
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam([
        {'params': model.conv_params.parameters(), 'lr': 0.002},
        {'params': model.pool_params.parameters(), 'lr': 0.005},
        {'params': model.head.parameters(), 'lr': 0.01}
    ])
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_rec': [], 'train_spec': [],
        'val_loss': [], 'val_acc': [], 'val_rec': [], 'val_spec': []
    }
    
    print(f"\n--- Démarrage de l'entraînement LD-QCNN (k={TARGET_K}, qubits={NUM_QUBITS}) ---")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses, batch_accs, batch_recs, batch_specs = [], [], [], []
        
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels.to(logits.dtype))
            
            loss.backward()
            optimizer.step()
            
            acc, rec, spec = calculate_metrics(logits, labels)
            
            batch_losses.append(loss.item())
            batch_accs.append(acc)
            if not math.isnan(rec): batch_recs.append(rec)
            if not math.isnan(spec): batch_specs.append(spec)
            
            if (i+1) % 10 == 0:  # Modifié pour afficher plus souvent sur les petits batchs
                print(f"    Batch {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
        epoch_loss = np.mean(batch_losses)
        epoch_acc = np.mean(batch_accs)
        epoch_rec = np.mean(batch_recs) if batch_recs else 0.0
        epoch_spec = np.mean(batch_specs) if batch_specs else 0.0
        
        model.eval()
        val_losses, val_accs, val_recs, val_specs = [], [], [], []
        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_logits = model(val_images)
                v_loss = criterion(val_logits, val_labels.to(val_logits.dtype))
                
                v_acc, v_rec, v_spec = calculate_metrics(val_logits, val_labels)
                
                val_losses.append(v_loss.item())
                val_accs.append(v_acc)
                if not math.isnan(v_rec): val_recs.append(v_rec)
                if not math.isnan(v_spec): val_specs.append(v_spec)
                
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = np.mean(val_accs)
        epoch_val_rec = np.mean(val_recs) if val_recs else 0.0
        epoch_val_spec = np.mean(val_specs) if val_specs else 0.0
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"      -> Train - Acc: {epoch_acc:.4f} | Rappel: {epoch_rec:.4f} | Spec: {epoch_spec:.4f}")
        print(f"      -> Val   - Acc: {epoch_val_acc:.4f} | Rappel: {epoch_val_rec:.4f} | Spec: {epoch_val_spec:.4f}")

    print(f"\nTemps total d'exécution : {time.time() - start_time:.2f} secondes")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), history['train_acc'], 'g', label='Training acc')
    plt.plot(range(1, EPOCHS + 1), history['val_acc'], 'black', label='Validation acc')
    plt.title(f'Accuracy (LD-QCNN k={TARGET_K})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], 'g', label='Training loss')
    plt.plot(range(1, EPOCHS + 1), history['val_loss'], 'black', label='Validation loss')
    plt.title('Loss (BCEWithLogits)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. VISUALISATION DU CIRCUIT
# ============================================================================
def afficher_diagramme_circuit():
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
    )(dummy_input, model.conv_params, model.pool_params, model.active_wires_architecture, k_val, depth)
    
    fig.set_size_inches(32, 12)
    plt.title(f"Architecture LD-QCNN ({NUM_QUBITS} qubits, k={k_val}, depth={depth})", fontsize=14, pad=20)
    fig.savefig("circuit_ld_qcnn_k2.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
    afficher_diagramme_circuit()