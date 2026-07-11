import optuna
import glob
import pathlib
import json
import numpy as np
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF_MAX_HITS = 113.0

def numpy_gp_predict(X_train, y_train, X_test, length_scale=200.0, sigma_f=0.5, noise=0.08):
    N, D = X_train.shape
    M = X_test.shape[0]
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
        try:
            K_inv = np.linalg.inv(K)
            y_pred = np.dot(np.dot(K_s, K_inv), y_train)
            sigma = np.sqrt(np.maximum(1e-8, K_ss - np.sum(np.dot(K_s, K_inv) * K_s, axis=1, keepdims=True)))
            return y_pred.flatten(), sigma.flatten()
        except Exception:
            return np.ones(M) * np.mean(y_train), np.ones(M) * sigma_f

def main():
    # Recursively find all db files in studies folder
    db_paths = glob.glob(str(ROOT / "studies" / "**" / "*.db"), recursive=True)
    results = []
    
    # Load deflector log points for I term estimation
    reg_pts_all = []
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
                            if abs(float(ox)) <= 12.0 and abs(float(oy)) <= 12.0:
                                reg_pts_all.append({"ox": float(ox), "oy": float(oy), "v15": float(v15), "v18": float(v18)})
                except Exception:
                    pass

    for db in db_paths:
        rel_path = pathlib.Path(db).relative_to(ROOT)
        
        try:
            summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{db}")
        except Exception:
            continue
            
        for s in summaries:
            try:
                study = optuna.load_study(study_name=s.study_name, storage=f"sqlite:///{db}")
            except Exception:
                continue
                
            completed = [t for t in study.trials if t.value is not None and t.state == optuna.trial.TrialState.COMPLETE]
            if not completed:
                continue
                
            # 1. G Term
            G = 1.0
            cols = sorted([k for k in completed[0].params.keys()])
            X_train = np.array([[t.params[k] for k in cols] for t in completed])
            y_train = np.array([t.value for t in completed])
            
            np.random.seed(42)
            X_rand = np.random.uniform(-1000, 1000, size=(100, len(cols)))
            preds, sigmas = numpy_gp_predict(X_train, y_train, X_rand)
            if np.any(preds < -0.05) or np.any(sigmas < 0.0):
                G = 0.5
                
            # 2. C Term
            best_t = min(completed, key=lambda t: t.value)
            best_hits = best_t.user_attrs.get("simion_hits") or 0
            C = min(1.0, float(best_hits) / REF_MAX_HITS)
            
            # 3. A Term
            pts_hits = [t for t in completed if (t.user_attrs.get("simion_hits") or 0) > 0 and t.user_attrs.get("rk4_score") is not None]
            if len(pts_hits) >= 3:
                x_rk4 = np.array([t.user_attrs.get("rk4_score") for t in pts_hits])
                y_sim = np.array([t.value for t in pts_hits])
                rho, _ = spearmanr(x_rk4, y_sim)
                A = max(0.0, float(rho)) if np.isfinite(rho) else 0.0
            else:
                A = 0.0
                
            # 4. I Term
            reg_pts = []
            for p in reg_pts_all:
                for ct in completed:
                    if abs(ct.params.get("V15", 999) - p["v15"]) < 1.0 and abs(ct.params.get("V18", 999) - p["v18"]) < 1.0:
                        reg_pts.append(p)
                        break
            if len(reg_pts) >= 4:
                ox_v = np.array([p["ox"] for p in reg_pts])
                oy_v = np.array([p["oy"] for p in reg_pts])
                v15_v = np.array([p["v15"] for p in reg_pts])
                v18_v = np.array([p["v18"] for p in reg_pts])
                r15 = np.corrcoef(ox_v, v15_v)[0, 1]**2 if np.std(ox_v) > 0 and np.std(v15_v) > 0 else 0.0
                r18 = np.corrcoef(oy_v, v18_v)[0, 1]**2 if np.std(oy_v) > 0 and np.std(v18_v) > 0 else 0.0
                I = float(0.5 * (r15 + r18))
            else:
                I = float(max(0.0, 1.0 - (abs(best_t.params.get("V15", 0.0)) + abs(best_t.params.get("V18", 0.0))) / 1000.0))
                
            # 5. E Term
            sigmas_far = []
            for pt in X_rand:
                dists = np.sqrt(np.sum((X_train - pt)**2, axis=1))
                if np.min(dists) > 300.0:
                    _, sig = numpy_gp_predict(X_train, y_train, np.array([pt]))
                    sigmas_far.append(sig[0])
            E = float(np.mean(sigmas_far) / 0.5) if sigmas_far else 0.85
            E = min(1.0, max(0.0, E))
            
            # 6. D Term
            neighbors = []
            best_x = np.array([best_t.params[k] for k in cols])
            for t in completed:
                if t.number == best_t.number:
                    continue
                x_t = np.array([t.params[k] for k in cols])
                dist = np.linalg.norm(best_x - x_t)
                if dist < 250.0:
                    neighbors.append((x_t, t.value))
            if len(neighbors) >= 2:
                dx_sim = np.zeros(len(cols))
                dy_sim = 0.0
                for x_n, y_n in neighbors:
                    dx_sim += (x_n - best_x)
                    dy_sim += (y_n - best_t.value)
                if np.linalg.norm(dx_sim) > 0:
                    grad_sim = (dy_sim / np.linalg.norm(dx_sim)) * (dx_sim / np.linalg.norm(dx_sim))
                else:
                    grad_sim = np.zeros(len(cols))
                eps = 1e-4
                grad_gp = np.zeros(len(cols))
                for idx, c in enumerate(cols):
                    best_x_plus = best_x.copy()
                    best_x_plus[idx] += eps
                    y_plus, _ = numpy_gp_predict(X_train, y_train, np.array([best_x_plus]))
                    grad_gp[idx] = (y_plus[0] - best_t.value) / eps
                denom = np.linalg.norm(grad_gp) * np.linalg.norm(grad_sim)
                cos_sim = np.dot(grad_gp, grad_sim) / denom if denom > 0 else 0.0
                D = max(0.0, float(cos_sim))
            else:
                D = 0.75
                
            # 7. F Term
            first_30_trial = 60.0
            for t in sorted(completed, key=lambda t: t.number):
                h = t.user_attrs.get("simion_hits") or 0
                if h >= 30:
                    first_30_trial = float(t.number)
                    break
            F = float(1.0 - (first_30_trial / 60.0))
            
            # Score
            score = float(G * (0.30*C + 0.20*A + 0.15*I + 0.15*E + 0.10*D + 0.10*F))
            
            results.append({
                "path": str(rel_path),
                "study": s.study_name,
                "score": score,
                "hits": best_hits,
                "metrics": {"G": G, "C": C, "A": A, "I": I, "E": E, "D": D, "F": F}
            })
            
    # Sort and rank
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    
    print("======================================================================")
    print("      TOP 5 MEJORES MODELOS DETECTADOS EN EL DIRECTORIO STUDIES")
    print("======================================================================\n")
    
    for idx, r in enumerate(results[:5]):
        print(f"[{idx+1}] SCORE: {r['score']:.5f} | Modelo: {r['study']} | Hits: {r['hits']}")
        print(f"    Ruta de Archivo: {r['path']}")
        m = r['metrics']
        print(f"    Desglose: G={m['G']:.1f}, C={m['C']:.3f}, A={m['A']:.3f}, I={m['I']:.3f}, E={m['E']:.3f}, D={m['D']:.3f}, F={m['F']:.3f}")
        print("-" * 70)

if __name__ == "__main__":
    main()
