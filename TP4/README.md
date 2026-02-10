# TP Bundle Adjustment avec PyCeres

Ce TP implémente un pipeline complet de Bundle Adjustment pour optimiser simultanément les poses de caméras et les positions de points 3D à partir de correspondances 2D-3D.

## Structure des Fichiers

```
.
├── bundle_adjustment.py          # Implémentation principale du Bundle Adjustment
├── kitti_bundle_adjustment.py    # Pipeline complet pour données KITTI
├── REPONSES_QUESTIONS.md          # Réponses détaillées aux questions conceptuelles
└── README.md                      # Ce fichier
```

## Installation des Dépendances

```bash
pip install numpy opencv-python scipy matplotlib PyCeres
```

## Utilisation

### 1.

```bash
python bundle_adjustment.py
```

### 2. Pipeline Complet KITTI

```bash
python kitti_bundle_adjustment.py
```

**Structure attendue des données** :
```
data/Sequence/
│── 000358.png
│── 000359.png
│── ...
```

Ce pipeline :
1. Charge les images de la séquence
2. Détecte et apparie les features (ORB)
3. Triangule les points 3D
4. Exécute le Bundle Adjustment
5. Visualise les résultats


### Step 1: Modèle de Projection

Implémentation du modèle pinhole

```python
def project_point(X_camera, K):
    """
    u = fx * (Xc / Zc) + cu
    v = fy * (Yc / Zc) + cv
    """
```

### Step 2: Définition du Résidu (Fonction de Coût)

```python
class ReprojectionError(PyCeres.CostFunction):
    """
    Résidu : r_ij = p_observed_ij - π(R_j * X_w,i + t_j)
    
    Paramètres à optimiser :
    - camera_pose : [rx, ry, rz, tx, ty, tz] (6 DOF)
    - point_3d : [X, Y, Z] (3 DOF)
    """
```

### Step 3: Construction du Problème

```python
problem = PyCeres.Problem()

# Pour chaque observation (point i observé par caméra j)
for cam_idx, point_idx, u, v in observations:
    cost_function = ReprojectionError(u, v, K)
    problem.AddResidualBlock(cost_function, loss, 
                            camera_params[cam_idx], 
                            point_params[point_idx])

# Gauge fixing : fixer la première caméra
problem.SetParameterBlockConstant(camera_params[0])
```

### Step 4: Optimisation

```python
options = PyCeres.SolverOptions()
options.linear_solver_type = PyCeres.LinearSolverType.SPARSE_SCHUR
options.max_num_iterations = 50

summary = PyCeres.Summary()
PyCeres.Solve(options, problem, summary)
```

## Questions 
### Question 1: Robust Loss Function (Huber Loss)

### Effet de la fonction de perte de Huber sur les outliers

La fonction de perte de Huber est définie comme :

```
ρ(r) = { r²/2           si |r| ≤ δ
       { δ(|r| - δ/2)   si |r| > δ
```

**Comportement :**

1. **Pour les petites erreurs (|r| ≤ δ)** : 
   - Comportement quadratique (comme les moindres carrés classiques)
   - Optimisation efficace pour les bonnes correspondances

2. **Pour les grandes erreurs (|r| > δ)** :
   - Comportement linéaire
   - Réduit l'influence des outliers
   - Les mauvaises correspondances ont moins d'impact sur l'optimisation

**Avantages :**
- Robustesse aux fausses correspondances de points
- Évite que quelques outliers ne dominent l'optimisation
- Meilleure convergence en présence de bruit

**Dans le code :**
```python
loss = PyCeres.HuberLoss(1.0)  # δ = 1.0 pixel
```

Le paramètre δ = 1.0 signifie que les erreurs > 1 pixel seront traitées linéairement.

---

### Question 2: Calibration - Matrice Intrinsèque Inconnue

Si la matrice intrinsèque K est inconnue, il faut optimiser les paramètres intrinsèques 
en plus des poses et des points 3D.

### Paramètres à ajouter :

1. **fx, fy** : Longueurs focales en x et y
2. **cx, cy** : Point principal (centre optique)
3. **k1, k2, p1, p2** : Coefficients de distorsion (optionnel)

### Modification de la fonction de coût :

```python
class ReprojectionErrorWithIntrinsics(PyCeres.CostFunction):
    def __init__(self, observed_x, observed_y):
        super().__init__()
        self.observed = np.array([observed_x, observed_y])
        
        # 2 residuals, 3 parameter blocks:
        # - camera_pose: 6 DOF
        # - point_3d: 3 DOF
        # - intrinsics: 4 DOF (fx, fy, cx, cy)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([6, 3, 4])
    
    def Evaluate(self, parameters, residuals, jacobians):
        camera_pose = parameters[0]
        point_3d = parameters[1]
        intrinsics = parameters[2]  # [fx, fy, cx, cy]
        
        fx, fy, cx, cy = intrinsics
        
        # ... transformation comme avant ...
        
        # Projection avec paramètres intrinsèques variables
        u = fx * (Xc / Zc) + cx
        v = fy * (Yc / Zc) + cy
        
        residuals[0] = u - self.observed[0]
        residuals[1] = v - self.observed[1]
        
        return True
```

### Ajout au problème :

```python
# Initialisation des paramètres intrinsèques
intrinsic_params = np.array([718.8, 718.8, 607.1, 185.2])

# Pour chaque observation
problem.AddResidualBlock(
    cost_function,
    loss,
    camera_params[cam_idx],
    point_params[point_idx],
    intrinsic_params  # Nouveau bloc de paramètres
)
```
## Résultats Attendus

1. **Convergence** : La fonction de coût diminue à chaque itération
2. **Erreurs de reprojection** : Médiane < 1 pixel pour de bonnes données
3. **Trajectoire lissée** : Les poses de caméras deviennent plus cohérentes
4. **Nuage de points** : Structure 3D de la scène

## Notes

### Différences Python vs C++

Ce TP est en Python :

**Avantages Python** :
- Prototypage rapide
- Visualisation facile (matplotlib)
- Manipulation de données (numpy)

**Équivalent C++** :
- Ceres Solver natif (plus rapide)
- Eigen pour l'algèbre linéaire
- OpenCV pour la vision

### Performance

Pour de **grandes scènes** (>1000 caméras, >100k points) :
- Utiliser C++ avec Ceres natif
- Utiliser GPU pour certains calculs (feature matching)

