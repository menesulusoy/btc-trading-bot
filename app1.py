# pip install flask flask-socketio ccxt pandas numpy scikit-learn xgboost lightgbm joblib

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO
import ccxt, pandas as pd, numpy as np
import threading, time, joblib, os
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Sabitler ──────────────────────────────────────────────
FEATURES   = ['SMA_20','SMA_50','RSI','MACD','MACD_Hist',
               'BB_Width','BB_Pos','ATR_14','Hacim_Ratio','Getiri_%','Getiri_3h']
CONFIDENCE    = 0.58
STRONG_CONFIDENCE = 0.70
BASLANGIC     = 10_000
KOMISYON      = 0.0001
POZISYON_ORAN = 0.80
STOP_LOSS     = 0.015
TAKE_PROFIT   = 0.030
MIN_BEKLE     = 6

# Jupyter Notebook'ta eğitip kaydettiğin model dosyası.
# Bu dosyayı app1.py ile aynı klasöre koymalısın.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'btc_hybrid_model.pkl')

state = {
    "model": None, "model_ready": False, "accuracy": 0,
    "candles": [], "balance_usdt": BASLANGIC, "balance_btc": 0.0,
    "alis_fiyat": 0.0, "son_islem_mum_zamani": None,
    "signals": [], "trades": [], "win": 0, "lose": 0,
}

def add_features(df):
    d = df.copy()
    d['SMA_20']  = d['Kapanis'].rolling(20).mean()
    d['SMA_50']  = d['Kapanis'].rolling(50).mean()
    d['EMA_12']  = d['Kapanis'].ewm(span=12, adjust=False).mean()
    d['EMA_26']  = d['Kapanis'].ewm(span=26, adjust=False).mean()
    delta = d['Kapanis'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    d['RSI']         = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    d['MACD']        = d['EMA_12'] - d['EMA_26']
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist']   = d['MACD'] - d['MACD_Signal']
    bb_mid = d['Kapanis'].rolling(20).mean()
    bb_std = d['Kapanis'].rolling(20).std()
    d['BB_Upper'] = bb_mid + 2 * bb_std
    d['BB_Lower'] = bb_mid - 2 * bb_std
    d['BB_Width'] = (d['BB_Upper'] - d['BB_Lower']) / bb_mid
    d['BB_Pos']   = (d['Kapanis'] - d['BB_Lower']) / (d['BB_Upper'] - d['BB_Lower'] + 1e-9)
    tr = pd.concat([
        d['Yuksek'] - d['Dusuk'],
        (d['Yuksek'] - d['Kapanis'].shift()).abs(),
        (d['Dusuk']  - d['Kapanis'].shift()).abs()
    ], axis=1).max(axis=1)
    d['ATR_14']       = tr.rolling(14).mean()

    # Gerçek Stochastic %K hesaplaması.
    # Önceki sürümde dashboard'a Stochastic diye RSI gönderiliyordu; bu yanıltıcıydı.
    low_14 = d['Dusuk'].rolling(14).min()
    high_14 = d['Yuksek'].rolling(14).max()
    d['Stoch_K'] = 100 * (d['Kapanis'] - low_14) / (high_14 - low_14 + 1e-9)

    d['Hacim_SMA_20'] = d['Hacim'].rolling(20).mean()
    d['Hacim_Ratio']  = d['Hacim'] / (d['Hacim_SMA_20'] + 1e-9)
    d['Getiri_%']     = d['Kapanis'].pct_change() * 100
    d['Getiri_3h']    = d['Kapanis'].pct_change(3) * 100
    d['Hedef']        = (d['Kapanis'].shift(-1) > d['Kapanis']).astype(int)
    return d

def load_saved_model():
    """Jupyter Notebook'ta eğitilip kaydedilen modeli yükler.

    Bu sürüm doğruluğu app içinde yeniden hesaplamaz.
    Doğruluk değeri iki temiz yoldan okunabilir:
    1) Model paketi dict ise: loaded['test_accuracy_pct']
    2) Model nesnesinin içine eklenmişse: loaded.test_accuracy_pct

    Uygulama model doğruluğunu yeniden hesaplamaz. Jupyter'da kaydedilmiş
    btc_hybrid_model.pkl dosyasındaki değeri okuyarak dashboard'a gönderir.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            mesaj = (
                "❌ Model dosyası bulunamadı: btc_hybrid_model.pkl | "
                "Notebook'ta modeli kaydeden hücreyi çalıştır."
            )
            print(mesaj)
            socketio.emit('price_update', {"model_status": mesaj, "model_ready": False})
            state['model_ready'] = False
            state['accuracy'] = 0
            return

        loaded = joblib.load(MODEL_PATH)

        model = None
        test_accuracy = None
        train_accuracy = None
        created_at = None
        kayit_features = None

        
        if isinstance(loaded, dict) and 'model' in loaded:
            model = loaded['model']
            test_accuracy = loaded.get('test_accuracy_pct')
            train_accuracy = loaded.get('train_accuracy_pct')
            created_at = loaded.get('created_at')
            kayit_features = loaded.get('features')

        
        else:
            model = loaded
            test_accuracy = getattr(loaded, 'test_accuracy_pct', None)
            train_accuracy = getattr(loaded, 'train_accuracy_pct', None)
            created_at = getattr(loaded, 'created_at', None)
            kayit_features = getattr(loaded, 'features', None)

        if model is None or not hasattr(model, 'predict_proba'):
            mesaj = (
                "❌ Yüklenen dosyada kullanılabilir model bulunamadı. | "
                "Notebook'ta modeli tekrar kaydet."
            )
            print(mesaj)
            socketio.emit('price_update', {"model_status": mesaj, "model_ready": False, "accuracy": 0})
            state['model_ready'] = False
            state['accuracy'] = 0
            return

        if test_accuracy is None:
            mesaj = (
                "❌ Model dosyası yüklendi ama içinde test_accuracy_pct yok. | "
                "Notebook'ta modeli doğruluk bilgisiyle kaydeden hücreyi çalıştırmalısın. "
                "Dashboard doğruluğu yeniden hesaplamayacak."
            )
            print(mesaj)
            socketio.emit('price_update', {"model_status": mesaj, "model_ready": False, "accuracy": 0})
            state['model_ready'] = False
            state['accuracy'] = 0
            return

        test_accuracy = round(float(test_accuracy), 2)

        state['model'] = model
        state['model_ready'] = True
        state['accuracy'] = test_accuracy

        if kayit_features and list(kayit_features) != FEATURES:
            print("⚠️ Uyarı: Kaydedilen modelin FEATURES listesi app içindeki FEATURES ile birebir aynı değil.")
            print("Model FEATURES:", kayit_features)
            print("App FEATURES  :", FEATURES)

        print("=" * 60)
        print("✅ Jupyter modeli yüklendi")
        print(f"📦 Model dosyası      : {MODEL_PATH}")
        print(f"📊 Notebook test doğruluğu: %{state['accuracy']:.2f}")
        if train_accuracy is not None:
            print(f"📊 Eğitim doğruluğu   : %{float(train_accuracy):.2f}")
        if created_at:
            print(f"🕒 Model kayıt zamanı : {created_at}")
        print("ℹ️  Bu değer app içinde hesaplanmadı; doğrudan model dosyasından okundu.")
        print("=" * 60)

        mesaj = f"✅ Jupyter modeli yüklendi — Notebook Test Doğruluğu: %{state['accuracy']:.2f}"
        socketio.emit('price_update', {
            "model_status": mesaj,
            "model_ready": True,
            "accuracy": state['accuracy']
        })

    except Exception as e:
        mesaj = f"❌ Model yüklenirken hata oluştu: {e}"
        print(mesaj)
        socketio.emit('price_update', {"model_status": mesaj, "model_ready": False, "accuracy": 0})
        state['model_ready'] = False
        state['accuracy'] = 0

def live_loop():
    exc = ccxt.binance()
    dongu_sayaci = 0
    while True:
        try:
            bars = exc.fetch_ohlcv('BTC/USDT', '1h', limit=120)
            df_c = pd.DataFrame(bars, columns=['Tarih','Acilis','Yuksek','Dusuk','Kapanis','Hacim'])
            df_c['Tarih'] = pd.to_datetime(df_c['Tarih'], unit='ms') + pd.Timedelta(hours=3)
            df_c.set_index('Tarih', inplace=True)
            df_c = add_features(df_c).dropna(subset=FEATURES)
            
            if df_c.empty:
                time.sleep(5); continue

            last  = df_c.iloc[-1]
            price = float(last['Kapanis'])
            suan = datetime.now().strftime('%H:%M:%S')
            
            guncel_mum_zamani = df_c.index[-1]
            bekleme_tamam = (
               state["son_islem_mum_zamani"] is None or
               (guncel_mum_zamani - state["son_islem_mum_zamani"]) >= pd.Timedelta(hours=MIN_BEKLE)
            )

            candle = {
                "time":  df_c.index[-1].strftime('%H:%M'),
                "open":  float(last['Acilis']), "high":  float(last['Yuksek']),
                "low":   float(last['Dusuk']),  "close": price
            }
            if not state['candles'] or state['candles'][-1]['time'] != candle['time']:
                state['candles'].append(candle)
                if len(state['candles']) > 100: state['candles'].pop(0)

            rsi   = float(last['RSI'])    if not np.isnan(last['RSI'])    else 0
            atr   = float(last['ATR_14']) if not np.isnan(last['ATR_14']) else 0
            stoch = float(last['Stoch_K']) if not np.isnan(last['Stoch_K']) else 0

            bot_status = {"model_ready": state['model_ready'], "accuracy": state['accuracy'],
                          "rsi": rsi, "atr": atr, "stoch_k": stoch, "rise_probability": 0,
                          "accuracy_source": "Notebook test setinden okundu"}

            signal_data = None
            new_trade = None 

            if state['model_ready']:
                prob     = float(state['model'].predict_proba(df_c[FEATURES].iloc[[-1]])[0][1])
                prob_pct = round(prob * 100, 1)
                bot_status['rise_probability'] = prob_pct

                # 🟢 ALIM MANTIĞI
                normal_al = prob >= CONFIDENCE and bekleme_tamam
                guclu_al = prob >= STRONG_CONFIDENCE

                if state['balance_btc'] == 0 and (normal_al or guclu_al):
                    harcama  = state['balance_usdt'] * POZISYON_ORAN
                    komisyon = harcama * KOMISYON
                    state['balance_btc']    = (harcama - komisyon) / price
                    state['balance_usdt']  -= harcama
                    state['alis_fiyat']     = price
                    state["son_islem_mum_zamani"] = guncel_mum_zamani
                    
                    trade       = {"type":"AL","time":suan,"price":price,"amount":round(state['balance_btc'],6),"pnl":None}
                    signal_data = {"type":"AL","reason":f"Model yükseliş skoru %{prob_pct}","time":suan,"price":price,"confidence":prob}
                    
                    new_trade = trade 
                    state['trades'].insert(0, trade)
                    state['signals'].insert(0, signal_data)
                    print(f"🟢 AL — Saat: {suan} | Fiyat: ${price:,.2f}")

                # 🔴 SATIŞ MANTIĞI
                elif state['balance_btc'] > 0:
                    degisim     = (price - state['alis_fiyat']) / state['alis_fiyat']
                    should_sell = (prob < CONFIDENCE and bekleme_tamam) \
                                  or degisim <= -STOP_LOSS or degisim >= TAKE_PROFIT
                                  
                    if should_sell:
                        komisyon = state['balance_btc'] * price * KOMISYON
                        gelir    = state['balance_btc'] * price - komisyon
                        pnl      = gelir - (state['alis_fiyat'] * state['balance_btc'])
                        
                        if pnl > 0:
                           state['win'] += 1
                        else:
                           state['lose'] += 1
                        
                        tip    = 'SAT-SL' if degisim <= -STOP_LOSS else ('SAT-TP' if degisim >= TAKE_PROFIT else 'SAT')
                        reason = "Stop-Loss" if tip=='SAT-SL' else ("Take-Profit" if tip=='SAT-TP' else f"Model skoru eşiğin altına düştü %{prob_pct}")
                        
                        # 📌 Satış gerçekleştiğinde, AL işleminin hareketli PnL'ini sıfırlayıp çizgi (-) yapıyoruz.
                        if len(state['trades']) > 0 and state['trades'][0]['type'] == 'AL':
                            state['trades'][0]['pnl'] = None
                            
                        trade       = {"type":tip,"time":suan,"price":price,"amount":round(state['balance_btc'],6),"pnl":round(pnl,2)}
                        signal_data = {"type":tip,"reason":reason,"time":suan,"price":price,"confidence":1-prob}
                        
                        new_trade = trade 
                        state['trades'].insert(0, trade)
                        state['signals'].insert(0, signal_data)
                        
                        state['balance_usdt'] += gelir
                        state['balance_btc']   = 0.0
                        state["son_islem_mum_zamani"] = guncel_mum_zamani
                        print(f"🔴 {tip} — Saat: {suan} | PnL: ${pnl:+.2f}")
                    else:
                        if dongu_sayaci % 3 == 0:
                            reason = f"Pozisyon korunuyor (Anlık: %{(degisim*100):+.2f})"
                            signal_data = {"type":"HOLD","reason":reason,"time":suan,"price":price,"confidence":prob}
                            state['signals'].insert(0, signal_data)
                else:
                    if dongu_sayaci % 3 == 0:
                        reason = f"Alım eşiği altında, izleniyor"
                        signal_data = {"type":"HOLD","reason":reason,"time":suan,"price":price,"confidence":prob}
                        state['signals'].insert(0, signal_data)

            # 📌 HAREKETLİ PNL (UNREALIZED PNL) HESAPLAMA:
            # İşlem açıldıktan sonra henüz satılmamış (Açık) bir pozisyon varsa
            if state['balance_btc'] > 0 and len(state['trades']) > 0 and state['trades'][0]['type'] == 'AL':
                anlik_deger = state['balance_btc'] * price
                maliyet = state['balance_btc'] * state['alis_fiyat']
                komisyon_tahmini = anlik_deger * KOMISYON  # Satarken ödenecek tahmini komisyon düşülür (Net Kâr için)
                canli_pnl = anlik_deger - maliyet - komisyon_tahmini
                
                # Arayüze göndermeden önce AL işleminin PnL bilgisini anlık güncelliyoruz
                state['trades'][0]['pnl'] = round(canli_pnl, 2)

            total_trades = state['win'] + state['lose']
            win_rate     = round(state['win'] / total_trades * 100) if total_trades > 0 else 0
            total_val    = state['balance_usdt'] + state['balance_btc'] * price
            pnl_total    = total_val - BASLANGIC

            portfolio = {
                "total_value":  round(total_val, 2), "balance_usdt": round(state['balance_usdt'], 2),
                "balance_btc":  state['balance_btc'], "win_rate":     win_rate,
                "total_trades": total_trades,        "pnl":          round(pnl_total, 2),
                "pnl_pct":      round(pnl_total / BASLANGIC * 100, 2),
            }

            socketio.emit('price_update', {
                "price":      price,                  "portfolio":  portfolio,
                "bot_status": bot_status,             "candle":     candle,
                "signals":    state['signals'][:20],  "trades":     state['trades'][:20],
            })

            if signal_data:
                socketio.emit('new_signal', {
                    "signal":     signal_data,        "portfolio":  portfolio,
                    "bot_status": bot_status,         "trade":      new_trade 
                })
            dongu_sayaci += 1
        except Exception as e:
            print("Live loop hatası:", e)
        time.sleep(5)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/init')
def api_init():
    exc    = ccxt.binance()
    ticker = exc.fetch_ticker('BTC/USDT')
    price  = ticker['last']
    stats  = {
        "change_pct": round(ticker['percentage'] or 0, 2),
        "high": ticker['high'], "low":  ticker['low'],
    }

    # İlk açılışta dashboard'a 0 değer göndermemek için son mumlardan indikatörleri hesaplıyoruz.
    bars = exc.fetch_ohlcv('BTC/USDT', '1h', limit=120)

    if len(state['candles']) < 10:
        state['candles'] = [{
            "time":  (pd.to_datetime(b[0], unit='ms') + pd.Timedelta(hours=3)).strftime('%H:%M'),
            "open":  b[1], "high":  b[2], "low":   b[3], "close": b[4]
        } for b in bars[-100:]]

    bot_status = {
        "model_ready": state['model_ready'],
        "accuracy": state['accuracy'],
        "accuracy_source": "Notebook test setinden okundu",
        "rsi": 0,
        "atr": 0,
        "stoch_k": 0,
        "rise_probability": 0
    }

    try:
        df_init = pd.DataFrame(bars, columns=['Tarih','Acilis','Yuksek','Dusuk','Kapanis','Hacim'])
        df_init['Tarih'] = pd.to_datetime(df_init['Tarih'], unit='ms') + pd.Timedelta(hours=3)
        df_init.set_index('Tarih', inplace=True)
        df_init = add_features(df_init).dropna(subset=FEATURES)

        if not df_init.empty:
            last = df_init.iloc[-1]
            bot_status['rsi'] = float(last['RSI']) if not np.isnan(last['RSI']) else 0
            bot_status['atr'] = float(last['ATR_14']) if not np.isnan(last['ATR_14']) else 0
            bot_status['stoch_k'] = float(last['Stoch_K']) if not np.isnan(last['Stoch_K']) else 0

            if state['model_ready']:
                prob = float(state['model'].predict_proba(df_init[FEATURES].iloc[[-1]])[0][1])
                bot_status['rise_probability'] = round(prob * 100, 1)
    except Exception as e:
        print("Init indikatör hesaplama hatası:", e)

    total_val = state['balance_usdt'] + state['balance_btc'] * price
    pnl       = total_val - BASLANGIC
    total_tr  = state['win'] + state['lose']
    
    portfolio = {
        "total_value":  round(total_val, 2), "balance_usdt": round(state['balance_usdt'], 2),
        "balance_btc":  state['balance_btc'], "win_rate":     round(state['win'] / total_tr * 100) if total_tr else 0,
        "total_trades": total_tr,            "pnl":          round(pnl, 2),
        "pnl_pct":      round(pnl / BASLANGIC * 100, 2),
    }
                  
    return jsonify({
        "price":         price,           "price_history": state['candles'],
        "portfolio":     portfolio,       "signals":       state['signals'][:20],
        "trades":        state['trades'][:20], "stats_24h":     stats,
        "bot_status":    bot_status,
    })

if __name__ == '__main__':
    # Jupyter Notebook'ta eğitilip kaydedilen btc_hybrid_model.pkl dosyası yüklenir.
    load_saved_model()
    threading.Thread(target=live_loop, daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)