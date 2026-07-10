import optuna
import pathlib
import json
import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDIES = ["gemelo_db_v2", "gemelo_v2_bender6d", "gemelo_v2_bender7d"]

# GP functions for E (extrapolation) and D (gradients) evaluation
def numpy_gp_predict(X_train, y_train, X_test, length_scale=200.0, sigma_f=0.5, noise=0.08):
    """Simple NumPy GP regressor with RBF kernel."""
    N, D = X_train.shape
    M = X_test.shape[0]
    
    # RBF Kernel matrices
    dist_train = np.sum(X_train**2, axis=1, keepdims=True) + np.sum(X_train**2, axis=1) - 2 * np.dot(X_train, X_train.T)
    K = sigma_f**2 * np.exp(-0.5 * dist_train / (length_scale**2)) + (noise**2) * np.eye(N)
    
    dist_test_train = np.sum(X_test**2, axis=1, keepdims=True) + np.sum(X_train**2, axis=1) - 2 * np.dot(X_test, X_train.T)
    K_s = sigma_f**2 * np.exp(-0.5 * dist_test_train / (length_scale**2))
    
    K_ss = sigma_f**2 * np.ones((M, 1)) + (noise**2)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        y_pred = np.dot(K_s, alpha)
        
        v = np.linalg.solve(L, K_s.T)
        sigma = np.sqrt(np.maximum(1e-8, K_ss - np.sum(v**2, axis=0, keepdims=True).T))
        return y_pred.flatten(), sigma.flatten()
    except np.linalg.LinAlgError:
        # Fallback to simple inverse
        try:
            K_inv = np.linalg.inv(K)
            y_pred = np.dot(np.dot(K_s, K_inv), y_train)
            sigma = np.sqrt(np.maximum(1e-8, K_ss - np.sum(np.dot(K_s, K_inv) * K_s, axis=1, keepdims=True)))
            return y_pred.flatten(), sigma.flatten()
        except Exception:
            return np.ones(M) * np.mean(y_train), np.ones(M) * sigma_f

def main():
    print("======================================================================")
    print("        EVALUACIÓN CIENTÍFICA DE GEMELOS DIGITALES (HACKATHON)")
    print("======================================================================\n")
    
    # Reference maximum hits found in clean runs
    REF_MAX_HITS = 113.0
    
    for study_name in STUDIES:
        db_path = ROOT / "studies" / f"{study_name}.db"
        if not db_path.exists():
            print(f"[-] Base de datos para {study_name} no encontrada. Saltando...\n")
            continue
            
        try:
            study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")
        except Exception as e:
            print(f"[-] Error cargando {study_name}: {e}\n")
            continue
            
        completed = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f"[-] No hay trials completados en {study_name}.\n")
            continue
            
        print(f"=== Analizando modelo: {study_name} ({len(completed)} trials) ===")
        
        # 1. COMPUERTA DE ADMISIBILIDAD (G)
        # Generate 200 random points and check for non-physical predictions
        G = 1.0
        # Parse params structure
        cols = sorted([k for k in completed[0].params.keys()])
        X_train = np.array([[t.params[k] for k in cols] for t in completed])
        y_train = np.array([t.value for t in completed])
        
        # Predict at random positions inside bounds
        np.random.seed(42)
        X_rand = np.random.uniform(-1000, 1000, size=(200, len(cols)))
        preds, sigmas = numpy_gp_predict(X_train, y_train, X_rand)
        if np.any(preds < -0.05) or np.any(sigmas < 0.0):
            G = 0.5  # Penalize non-physical scores
            print("  [G] Penalizacion por prediccion no fisica activa (G = 0.5)")
        else:
            print("  [G] Compuerta de admisibilidad aprobada (G = 1.0)")
            
        # 2. DIRECCIÓN (C)
        best_t = min(completed, key=lambda t: t.value)
        best_hits = best_t.user_attrs.get("simion_hits") or 0
        C = min(1.0, float(best_hits) / REF_MAX_HITS)
        print(f"  [C] Direccion (Hits del recomendado: {best_hits} vs Ref: {REF_MAX_HITS:.0f}): C = {C:.4f}")
        
        # 3. EXACTITUD EN REGIÓN INFORMATIVA (A)
        # Only trials with hits > 0
        pts_hits = [t for t in completed if (t.user_attrs.get("simion_hits") or 0) > 0 and t.user_attrs.get("rk4_score") is not None]
        if len(pts_hits) >= 3:
            x_rk4 = np.array([t.user_attrs.get("rk4_score") for t in pts_hits])
            y_sim = np.array([t.value for t in pts_hits])
            rho, _ = spearmanr(x_rk4, y_sim)
            A = max(0.0, float(rho)) if np.isfinite(rho) else 0.0
        else:
            A = 0.0
        print(f"  [A] Exactitud en region informativa (Spearman rho, n={len(pts_hits)}): A = {A:.4f}")
        
        # 4. CONSIGNA INVERSA (I)
        # Check deflector calibration linearity R^2 using registro_corridas.jsonl if possible
        I = 0.0
        reg_pts = []
        log_path = ROOT / "studies" / "registro_corridas.jsonl"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if d.get("fuente") == "simion" and d.get("features") is not None:
                            feats = d["features"]
                            volts = d["voltajes"]
                            ox = feats.get("offset_x_mm")
                            oy = feats.get("offset_y_mm")
                            v15 = volts.get("15") or volts.get("V15")
                            v18 = volts.get("18") or volts.get("V18")
                            hits = feats.get("hits") or 0
                            if all(val is not None for val in (ox, oy, v15, v18)) and float(hits) > 5:
                                # Check if these voltages belong to this model
                                for ct in completed:
                                    if abs(ct.params.get("V15", 999) - float(v15)) < 1.0 and abs(ct.params.get("V18", 999) - float(v18)) < 1.0:
                                        if abs(float(ox)) <= 12.0 and abs(float(oy)) <= 12.0:
                                            reg_pts.append({"ox": float(ox), "oy": float(oy), "v15": float(v15), "v18": float(v18)})
                                            break
                    except Exception:
                        pass
        
        if len(reg_pts) >= 4:
            ox_v = np.array([p["ox"] for p in reg_pts])
            oy_v = np.array([p["oy"] for p in reg_pts])
            v15_v = np.array([p["v15"] for p in reg_pts])
            v18_v = np.array([p["v18"] for p in reg_pts])
            # R^2 fit
            r15 = np.corrcoef(ox_v, v15_v)[0, 1]**2 if np.std(ox_v) > 0 and np.std(v15_v) > 0 else 0.0
            r18 = np.corrcoef(oy_v, v18_v)[0, 1]**2 if np.std(oy_v) > 0 and np.std(v18_v) > 0 else 0.0
            I = float(0.5 * (r15 + r18))
        else:
            # Fallback to a baseline score based on deflector proximity to 0
            I = float(max(0.0, 1.0 - (abs(best_t.params.get("V15", 0.0)) + abs(best_t.params.get("V18", 0.0))) / 1000.0))
        print(f"  [I] Consigna inversa (Linealidad R2 deflectores, n={len(reg_pts)}): I = {I:.4f}")
        
        # 5. EXTRAPOLACIÓN Y HONESTIDAD (E)
        # Evaluate sigmas in unvisited outer space (voltages far from data)
        sigmas_far = []
        for pt in X_rand:
            dists = np.sqrt(np.sum((X_train - pt)**2, axis=1))
            if np.min(dists) > 400.0:  # Far away points
                _, sig = numpy_gp_predict(X_train, y_train, np.array([pt]))
                sigmas_far.append(sig[0])
        E = float(np.mean(sigmas_far) / 0.5) if sigmas_far else 0.85
        E = min(1.0, max(0.0, E))
        print(f"  [E] Extrapolacion y honestidad (Incertidumbre GP lejana): E = {E:.4f}")
        
        # 6. DIFERENCIABILIDAD (D)
        # Cosine similarity between GP gradient and SIMION local differences
        # Find neighbors of best trial
        neighbors = []
        best_x = np.array([best_t.params[k] for k in cols])
        for t in completed:
            if t.number == best_t.number:
                continue
            x_t = np.array([t.params[k] for k in cols])
            dist = np.linalg.norm(best_x - x_t)
            if dist < 250.0:  # Nearby trial
                neighbors.append((x_t, t.value))
                
        if len(neighbors) >= 2:
            # Compute SIMION local slope
            dx_sim = np.zeros(len(cols))
            dy_sim = 0.0
            for x_n, y_n in neighbors:
                dx_sim += (x_n - best_x)
                dy_sim += (y_n - best_t.value)
            # Normalize
            if np.linalg.norm(dx_sim) > 0:
                grad_sim = (dy_sim / np.linalg.norm(dx_sim)) * (dx_sim / np.linalg.norm(dx_sim))
            else:
                grad_sim = np.zeros(len(cols))
                
            # GP Gradient estimation at best point (finite difference of GP)
            eps = 1e-4
            grad_gp = np.zeros(len(cols))
            for idx, c in enumerate(cols):
                best_x_plus = best_x.copy()
                best_x_plus[idx] += eps
                y_plus, _ = numpy_gp_predict(X_train, y_train, np.array([best_x_plus]))
                grad_gp[idx] = (y_plus[0] - best_t.value) / eps
                
            # Cosine similarity
            denom = np.linalg.norm(grad_gp) * np.linalg.norm(grad_sim)
            cos_sim = np.dot(grad_gp, grad_sim) / denom if denom > 0 else 0.0
            D = max(0.0, float(cos_sim))
        else:
            D = 0.75  # Default baseline for analytical differentiability of smooth GP
        print(f"  [D] Diferenciabilidad (Alineacion de gradiente GP-SIMION): D = {D:.4f}")
        
        # 7. EFICIENCIA DE DATOS (F)
        # Find first trial where hits >= 30
        first_30_trial = 60.0
        for t in sorted(completed, key=lambda t: t.number):
            h = t.user_attrs.get("simion_hits") or 0
            if h >= 30:
                first_30_trial = float(t.number)
                break
        F = float(1.0 - (first_30_trial / 60.0))
        print(f"  [F] Eficiencia de datos (Trial de primer hit >= 30: #{first_30_trial:.0f}): F = {F:.4f}")
        
        # FINAL SCORE CALCULATION
        # J_score = G * (0.30*C + 0.20*A + 0.15*I + 0.15*E + 0.10*D + 0.10*F)
        score = float(G * (0.30*C + 0.20*A + 0.15*I + 0.15*E + 0.10*D + 0.10*F))
        
        print(f"\n  ===> PUNTAJE DE EVALUACIÓN FINAL DEL GEMELO: {score:.5f} <===\n")
        print("-" * 70)

if __name__ == "__main__":
    main()
