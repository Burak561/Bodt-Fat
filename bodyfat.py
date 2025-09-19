# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 21:21:51 2025

@author: pc
"""

# run_bodyfat_short.py
# Gerekenler: pip install scikit-learn pandas numpy
# Not: bodyfat.csv ayni klasorde olmali (Kaggle: fedesoriano/body-fat-prediction-dataset)

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# --- 1) VERI VE MODEL (Linear Regression) ---
df = pd.read_csv("C:/Users/pc/Downloads/bodyfat.csv")
y = df["BodyFat"].values
X_all = df.drop(columns=["BodyFat"])
feats_all = X_all.columns.tolist()
feats_nodens = [c for c in feats_all if c != "Density"]

Xa_tr, Xa_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, random_state=42)
Xn_tr, Xn_te, y2_tr, y2_te = train_test_split(df[feats_nodens], y, test_size=0.2, random_state=42)

model_all    = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]).fit(Xa_tr, y_tr)
model_nodens = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]).fit(Xn_tr, y2_tr)

# --- 2) KULLANICIDAN DEĞER AL (kg/cm) ---
print("\n--- Ölçülerini gir (kg/cm). Boş bırakılabilen: Density ---")
age      = int(input("Yaş: ").strip())
sex      = input("Cinsiyet (Erkek/Kadin): ").strip() or "Erkek"
act      = input("Aktivite (hareketsiz/hafif/orta/yuksek/cok_yuksek): ").strip() or "orta"
w_kg     = float(input("Kilo (kg): ").replace(",", "."))
h_cm     = float(input("Boy (cm): ").replace(",", "."))
neck     = float(input("Boyun çevresi (cm): ").replace(",", "."))
chest    = float(input("Göğüs çevresi (cm): ").replace(",", "."))
abd      = float(input("Bel/Abdomen (cm): ").replace(",", "."))
hip      = float(input("Kalça (cm): ").replace(",", "."))
thigh    = float(input("Uyluk (cm): ").replace(",", "."))
knee     = float(input("Diz (cm): ").replace(",", "."))
ankle    = float(input("Ayak bileği (cm): ").replace(",", "."))
biceps   = float(input("Pazı (cm): ").replace(",", "."))
forearm  = float(input("Ön kol (cm): ").replace(",", "."))
wrist    = float(input("Bilek (cm): ").replace(",", "."))
dens_str = input("Density (opsiyonel, bilmiyorsan boş bırak): ").strip()
density  = float(dens_str.replace(",", ".")) if dens_str else None

# --- 3) BİRİM DÖNÜŞÜMÜ (dataset uyumu) ---
w_lb = w_kg * 2.20462
h_in = h_cm / 2.54

feats_all_vals = {
    "Density": density if density is not None else 1.05,  # yoksa varsayılan
    "Age": age, "Weight": w_lb, "Height": h_in, "Neck": neck, "Chest": chest,
    "Abdomen": abd, "Hip": hip, "Thigh": thigh, "Knee": knee, "Ankle": ankle,
    "Biceps": biceps, "Forearm": forearm, "Wrist": wrist
}
feats_nodens_vals = {k: v for k, v in feats_all_vals.items() if k != "Density"}

# --- 4) LINEER REGRESYONLA BODYFAT TAHMİN ---
if density is None:
    x = np.array([[feats_nodens_vals[f] for f in feats_nodens]], dtype=float)
    bf_pred = float(model_nodens.predict(x)[0])
else:
    x = np.array([[feats_all_vals[f] for f in feats_all]], dtype=float)
    bf_pred = float(model_all.predict(x)[0])

# sınıflandırma
if bf_pred < 10:    status = "Atletik (10% altı)"
elif bf_pred < 18:  status = "Sağlıklı (%10–%18)"
elif bf_pred <= 25: status = "Sınır bölge (%18–%25)"
else:               status = "Sağlıksız (>25%)"

print(f"\nTahmini BodyFat: {bf_pred:.1f}%  ->  {status}")

# --- 5) TDEE (Mifflin) + HEDEF KALORİ VE MAKROLAR (40/30/30) ---
sex_l = sex.lower()
bmr = (10*w_kg + 6.25*h_cm - 5*age + (5 if sex_l.startswith("e") else -161))
af = {"hareketsiz":1.2,"hafif":1.375,"orta":1.55,"yuksek":1.725,"cok_yuksek":1.9}.get(act.lower(),1.55)
tdee = bmr * af
target_kcal = max(1200, tdee*0.80)  # %20 açık (istenirse değiştir)
carb_g = round(0.40*target_kcal/4, 1)
prot_g = round(0.30*target_kcal/4, 1)
fat_g  = round(0.30*target_kcal/9, 1)

print(f"TDEE: {int(tdee)} kcal | Hedef (cut): {int(target_kcal)} kcal")
print(f"Makro hedefleri -> KH: {carb_g} g, PRO: {prot_g} g, YAĞ: {fat_g} g")

# --- 6) 100g BAZLI GIDA TABLOSU VE RASTGELE PLAN ---

foods = pd.DataFrame([
    ("Yulaf (kuru)",                389, 66.3, 16.9,  6.9),
    ("Pirinç (pişmiş)",             130, 28.0,  2.4,  0.3),
    ("Tavuk göğüs (pişmiş)",        165,  0.0, 31.0,  3.6),
    ("Yoğurt (sade, %3)",            61,  4.7,  3.5,  3.3),
    ("Zeytinyağı",                  884,  0.0,  0.0,100.0),
    ("Badem",                       579, 21.6, 21.2, 49.9),
    ("Yeşil mercimek (pişmiş)",     116, 20.1,  9.0,  0.4),
    ("Ton balık (suda)",            132,  0.0, 29.0,  1.0),
    ("Yumurta (pişmiş)",            155,  1.1, 13.0, 10.6),
    ("Elma",                         52, 14.0,  0.3,  0.2),
], columns=["food","kcal100","kh100","pro100","yag100"])

# Hedef değerler (tek sayı olmalı)
target_kcal = 2000
carb_g      = 250
prot_g      = 100
fat_g       = 70

# Diyet planını hesaplayan fonksiyon
def eval_plan(grams):
    f = foods
    factor = grams/100.0
    kcal = (f.kcal100*factor).sum()
    kh   = (f.kh100  *factor).sum()
    pro  = (f.pro100 *factor).sum()
    yag  = (f.yag100 *factor).sum()
    loss = ((kcal-target_kcal)/target_kcal)**2 + ((kh-carb_g)/carb_g)**2 \
         + ((pro-prot_g)/prot_g)**2 + ((yag-fat_g)/fat_g)**2
    return float(loss), kcal, kh, pro, yag  # loss tek sayı olarak döndürülüyor

# Rastgele optimizasyon
np.random.seed(1)
best = None
n = len(foods)
for _ in range(4000):   
    mask = (np.random.rand(n) > 0.4)
    hi = np.full(n, 400.0)
    hi[foods.food.to_numpy() == "Zeytinyağı"] = 40.0
    grams = np.where(mask, np.random.uniform(0, hi), 0.0)
    out = eval_plan(grams)
    if (best is None) or (out[0] < best[0][0]):
        best = (out, grams)

# Sonuçları tabloya aktar
(_, kcal, kh, pro, yag), grams = best
plan = foods.copy()
plan["gram"] = grams.round(1)
plan = plan[plan["gram"]>0]
plan["kcal"] = (plan.kcal100*plan.gram/100).round(1)
plan["KH_g"] = (plan.kh100  *plan.gram/100).round(1)
plan["PRO_g"]= (plan.pro100 *plan.gram/100).round(1)
plan["YAĞ_g"]= (plan.yag100 *plan.gram/100).round(1)

# Sonuçları yazdır
print("\n--- Günlük Diyet Planı (rastgele/100g ölçek) ---")
print(plan[["food","gram","kcal","KH_g","PRO_g","YAĞ_g"]].to_string(index=False))
print(f"\nTOPLAM: {int(plan.kcal.sum())} kcal | KH {plan.KH_g.sum():.1f} g | PRO {plan.PRO_g.sum():.1f} g | YAĞ {plan['YAĞ_g'].sum():.1f} g")
print(f"HEDEF:  {int(target_kcal)} kcal | KH {carb_g} g | PRO {prot_g} g | YAĞ {fat_g} g")