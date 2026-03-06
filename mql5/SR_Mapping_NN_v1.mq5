//+------------------------------------------------------------------+
//|                                          SR_Mapping_NN_v1.mq5    |
//|                        SR_Mapping_Foundation + Neural Network     |
//|                        Grid/Martingale REMOVED, ONNX Filter ON   |
//+------------------------------------------------------------------+
#property copyright "SR_Mapping_NN v1.0"
#property link      ""
#property version   "1.00"
#property strict

//--- ONNX includes
#include <Math\Stat\Math.mqh>

//--- Input Parameters: Entry Logic (dari SR_Mapping_Foundation v7.1)
input group "=== Entry Parameters ==="
input int    InpRSI_Period        = 14;        // RSI Period
input double InpRSI_Oversold      = 40;        // RSI Oversold (BUY threshold)
input double InpRSI_Overbought    = 70;        // RSI Overbought (SELL threshold)
input int    InpATR_Period        = 14;        // ATR Period
input int    InpATR_MinPoints     = 200;       // ATR Min Points (filter)
input int    InpATR_MaxPoints     = 2500;      // ATR Max Points (filter)
input int    InpSNR_Tolerance     = 50;        // S/R Tolerance (points)
input int    InpDailyRangeThresh  = 1000;      // Daily Range Threshold (points)
input int    InpFractalLookback   = 2000;      // Fractal Lookback Bars
input int    InpTradingHourStart  = 2;         // Trading Hour Start
input int    InpTradingHourEnd    = 18;        // Trading Hour End
input int    InpMaxSpreadPoints   = 400;       // Max Spread (points)
input int    InpMinBarsElapsed    = 10;        // Min Bars Between Trades (M30 bars = 5 hours)

input group "=== Neural Network Filter ==="
input bool   InpUseNNFilter       = true;      // Enable NN Entry Filter
input double InpConfidenceThresh  = 0.51;      // NN Confidence Threshold (0.0-1.0)
input string InpOnnxModelFile     = "sr_mapping_nn.onnx"; // ONNX Model Filename

input group "=== Risk Management ==="
input double InpLotSize           = 0.01;      // Lot Size
input bool   InpUseStopLoss       = true;      // Use Stop Loss (WAJIB)
input bool   InpUseTakeProfit     = true;      // Use Take Profit
input bool   InpUseATR_SLTP       = true;      // ATR-based SL/TP
input double InpSL_ATR_Mult       = 1.6;       // SL = ATR * Multiplier
input double InpTP_ATR_Mult       = 2.0;       // TP = ATR * Multiplier
input int    InpFixedSL_Points    = 5000;      // Fixed SL (if not ATR-based)
input int    InpFixedTP_Points    = 8000;      // Fixed TP (if not ATR-based)
input int    InpMaxOpenTrades     = 1;         // Max Open Trades

input group "=== Smart Trailing Stop ==="
input bool   InpUseSmartTrailing  = true;      // Enable Smart Trailing
input double InpTrail_ATR_Mult    = 5.0;       // Trail Trigger = ATR * Mult
input int    InpTrailStepPoints   = 300;       // Trail SL Step (points)
input int    InpMinSLMovement     = 50;        // Min SL Movement (anti-spam)

input group "=== EA Settings ==="
input int    InpMagicNumber       = 20250306;  // Magic Number
input string InpComment           = "SR_NN_v1"; // Order Comment

//--- Global Variables
int    g_handleRSI_M1 = INVALID_HANDLE;
int    g_handleATR_M1 = INVALID_HANDLE;
long   g_onnxHandle   = INVALID_HANDLE;
double g_currentResistance = 0;
double g_currentSupport    = 0;
double g_lastTradeTime     = 0;
double g_entryATR[];       // Store ATR at entry for trailing
bool   g_onnxLoaded = false;

//--- Point and symbol info
double g_point;
int    g_digits;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Symbol info
    g_point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    g_digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    
    //--- Create RSI indicator on M1
    g_handleRSI_M1 = iRSI(_Symbol, PERIOD_M1, InpRSI_Period, PRICE_CLOSE);
    if(g_handleRSI_M1 == INVALID_HANDLE)
    {
        Print("ERROR: Failed to create RSI indicator");
        return INIT_FAILED;
    }
    
    //--- Create ATR indicator on M1
    g_handleATR_M1 = iATR(_Symbol, PERIOD_M1, InpATR_Period);
    if(g_handleATR_M1 == INVALID_HANDLE)
    {
        Print("ERROR: Failed to create ATR indicator");
        return INIT_FAILED;
    }
    
    //--- Load ONNX model (if NN filter enabled)
    if(InpUseNNFilter)
    {
        LoadONNXModel();
    }
    
    //--- Initialize entry ATR storage
    ArrayResize(g_entryATR, InpMaxOpenTrades);
    ArrayInitialize(g_entryATR, 0);
    
    Print("SR_Mapping_NN v1 initialized successfully");
    Print("NN Filter: ", InpUseNNFilter ? "ENABLED" : "DISABLED");
    Print("Confidence Threshold: ", InpConfidenceThresh);
    Print("Point Size: ", g_point, " | Digits: ", g_digits);
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(g_handleRSI_M1 != INVALID_HANDLE) IndicatorRelease(g_handleRSI_M1);
    if(g_handleATR_M1 != INVALID_HANDLE) IndicatorRelease(g_handleATR_M1);
    if(g_onnxHandle != INVALID_HANDLE)   OnnxRelease(g_onnxHandle);
    
    Print("SR_Mapping_NN v1 deinitialized");
}

//+------------------------------------------------------------------+
//| Load ONNX Model                                                   |
//+------------------------------------------------------------------+
void LoadONNXModel()
{
    //--- Try to load from Files folder
    g_onnxHandle = OnnxCreate(InpOnnxModelFile, 
                              ONNX_DEFAULT);
    
    if(g_onnxHandle == INVALID_HANDLE)
    {
        //--- Try with full path
        string fullPath = TerminalInfoString(TERMINAL_DATA_PATH) + 
                          "\\MQL5\\Files\\" + InpOnnxModelFile;
        g_onnxHandle = OnnxCreate(fullPath, ONNX_DEFAULT);
    }
    
    if(g_onnxHandle == INVALID_HANDLE)
    {
        Print("WARNING: Could not load ONNX model '", InpOnnxModelFile, "'");
        Print("NN filter will be DISABLED. Error: ", GetLastError());
        g_onnxLoaded = false;
        return;
    }
    
    //--- Define input shape: [1, 26] (26 features)
    const long inputShape[] = {1, 26};
    if(!OnnxSetInputShape(g_onnxHandle, 0, inputShape))
    {
        Print("WARNING: Could not set ONNX input shape. Error: ", GetLastError());
        g_onnxLoaded = false;
        return;
    }
    
    //--- Define output shape
    const long outputShape[] = {1, 2};
    if(!OnnxSetOutputShape(g_onnxHandle, 1, outputShape))
    {
        Print("WARNING: Could not set ONNX output shape. Will try at runtime.");
    }
    
    g_onnxLoaded = true;
    Print("ONNX model loaded successfully: ", InpOnnxModelFile);
}

//+------------------------------------------------------------------+
//| BuildSNR — Fractal-based S/R Mapping                             |
//+------------------------------------------------------------------+
void BuildSNR()
{
    int lookback = MathMin(InpFractalLookback, Bars(_Symbol, PERIOD_M30));
    
    //--- Get fractal buffers
    int handleFractal = iFractals(_Symbol, PERIOD_M30);
    if(handleFractal == INVALID_HANDLE)
    {
        Print("ERROR: Failed to create Fractals indicator");
        return;
    }
    
    double fracUpper[], fracLower[];
    ArraySetAsSeries(fracUpper, true);
    ArraySetAsSeries(fracLower, true);
    
    CopyBuffer(handleFractal, 0, 0, lookback, fracUpper); // Upper fractal
    CopyBuffer(handleFractal, 1, 0, lookback, fracLower); // Lower fractal
    
    //--- Find nearest resistance (latest fractal UP)
    g_currentResistance = 0;
    for(int i = 0; i < lookback; i++)
    {
        if(fracUpper[i] != EMPTY_VALUE && fracUpper[i] > 0)
        {
            g_currentResistance = fracUpper[i];
            break;
        }
    }
    
    //--- Find nearest support (latest fractal DOWN)
    g_currentSupport = 0;
    for(int i = 0; i < lookback; i++)
    {
        if(fracLower[i] != EMPTY_VALUE && fracLower[i] > 0)
        {
            g_currentSupport = fracLower[i];
            break;
        }
    }
    
    IndicatorRelease(handleFractal);
}

//+------------------------------------------------------------------+
//| GetRSI_M1 — RSI value on M1 timeframe                           |
//+------------------------------------------------------------------+
double GetRSI_M1()
{
    double rsiBuffer[];
    ArraySetAsSeries(rsiBuffer, true);
    
    if(CopyBuffer(g_handleRSI_M1, 0, 0, 1, rsiBuffer) <= 0)
    {
        Print("ERROR: Failed to get RSI data");
        return 50.0; // neutral
    }
    
    return rsiBuffer[0];
}

//+------------------------------------------------------------------+
//| GetATR_M1 — ATR value on M1 timeframe (in points)               |
//+------------------------------------------------------------------+
double GetATR_M1_Points()
{
    double atrBuffer[];
    ArraySetAsSeries(atrBuffer, true);
    
    if(CopyBuffer(g_handleATR_M1, 0, 0, 1, atrBuffer) <= 0)
    {
        Print("ERROR: Failed to get ATR data");
        return 0;
    }
    
    return atrBuffer[0] / g_point;
}

//+------------------------------------------------------------------+
//| DetectDailyRange — Daily range direction                          |
//+------------------------------------------------------------------+
int DetectDailyRange()
{
    double dailyOpen = iOpen(_Symbol, PERIOD_D1, 0);
    double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double rangePoints = (currentBid - dailyOpen) / g_point;
    
    if(rangePoints >= InpDailyRangeThresh)
        return 1;   // BULLISH
    if(rangePoints <= -InpDailyRangeThresh)
        return -1;  // BEARISH
    
    return 0; // NO TRADE
}

//+------------------------------------------------------------------+
//| IsWithinTradingHours                                              |
//+------------------------------------------------------------------+
bool IsWithinTradingHours()
{
    MqlDateTime dt;
    TimeCurrent(dt);
    return (dt.hour >= InpTradingHourStart && dt.hour < InpTradingHourEnd);
}

//+------------------------------------------------------------------+
//| IsSpreadAcceptable                                                |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable()
{
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
    return (spread <= InpMaxSpreadPoints);
}

//+------------------------------------------------------------------+
//| IsMinBarsElapsed                                                  |
//+------------------------------------------------------------------+
bool IsMinBarsElapsed()
{
    if(g_lastTradeTime == 0) return true;
    
    double elapsed = (double)(TimeCurrent() - (datetime)g_lastTradeTime) / 60.0;
    return (elapsed >= InpMinBarsElapsed * 30); // M30 bars * 30 minutes
}

//+------------------------------------------------------------------+
//| CountOpenPositions                                                |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0)
        {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
            {
                count++;
            }
        }
    }
    return count;
}

//+------------------------------------------------------------------+
//| PrepareFeatures — Compute 26 features for NN                     |
//+------------------------------------------------------------------+
bool PrepareFeatures(float &features[])
{
    ArrayResize(features, 26);
    
    //--- Get indicator values
    double rsi = GetRSI_M1();
    double atrPoints = GetATR_M1_Points();
    double atr = atrPoints * g_point;
    double currentBid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double dailyOpen = iOpen(_Symbol, PERIOD_D1, 0);
    double dailyRange = currentBid - dailyOpen;
    int dailyDir = DetectDailyRange();
    
    //--- S/R distances
    double distRes = g_currentResistance - currentBid;
    double distSup = currentBid - g_currentSupport;
    double distResNorm = (atr > 0) ? distRes / atr : 0;
    double distSupNorm = (atr > 0) ? distSup / atr : 0;
    double srRange = g_currentResistance - g_currentSupport;
    double srPosition = (srRange > 0) ? (currentBid - g_currentSupport) / srRange : 0.5;
    
    //--- Time
    MqlDateTime dt;
    TimeCurrent(dt);
    int hour = dt.hour;
    int dow = dt.day_of_week - 1; // MT5 is 0=Sunday, convert to 0=Monday
    if(dow < 0) dow = 6;
    
    //--- Momentum (returns based on M30 close)
    double close_now = iClose(_Symbol, PERIOD_M30, 0);
    double close_3 = iClose(_Symbol, PERIOD_M30, 3);
    double close_5 = iClose(_Symbol, PERIOD_M30, 5);
    double close_10 = iClose(_Symbol, PERIOD_M30, 10);
    double close_20 = iClose(_Symbol, PERIOD_M30, 20);
    
    double ret3 = (close_3 > 0) ? ((close_now / close_3) - 1) * 100 : 0;
    double ret5 = (close_5 > 0) ? ((close_now / close_5) - 1) * 100 : 0;
    double ret10 = (close_10 > 0) ? ((close_now / close_10) - 1) * 100 : 0;
    double ret20 = (close_20 > 0) ? ((close_now / close_20) - 1) * 100 : 0;
    
    //--- ATR percentile (simplified: compare current ATR to recent)
    double atrArr[];
    ArraySetAsSeries(atrArr, true);
    CopyBuffer(g_handleATR_M1, 0, 0, 168, atrArr);
    int higher = 0;
    for(int i = 0; i < ArraySize(atrArr); i++)
        if(atrArr[i] <= atr) higher++;
    double atrPctile = (ArraySize(atrArr) > 0) ? (double)higher / ArraySize(atrArr) : 0.5;
    
    //--- ATR change
    double atr_5 = 0;
    if(ArraySize(atrArr) > 5)
        atr_5 = atrArr[5] * g_point; // Convert back to dollar value
    double atrChange = (atr_5 > 0) ? ((atr / atr_5) - 1) * 100 : 0;
    
    //--- RSI derived
    double rsiArr[];
    ArraySetAsSeries(rsiArr, true);
    CopyBuffer(g_handleRSI_M1, 0, 0, 10, rsiArr);
    double rsiSma = 0;
    for(int i = 0; i < MathMin(10, ArraySize(rsiArr)); i++)
        rsiSma += rsiArr[i];
    rsiSma /= MathMin(10, MathMax(1, ArraySize(rsiArr)));
    double rsiSlope = (ArraySize(rsiArr) >= 4) ? rsiArr[0] - rsiArr[3] : 0;
    
    //--- Near S/R
    double tolerance = atr * 0.20; // 20% of ATR
    int nearSupport = (MathAbs(distSup) <= tolerance) ? 1 : 0;
    int nearResistance = (MathAbs(distRes) <= tolerance) ? 1 : 0;
    
    //--- Candle patterns (from M30)
    double open_0 = iOpen(_Symbol, PERIOD_M30, 0);
    double high_0 = iHigh(_Symbol, PERIOD_M30, 0);
    double low_0 = iLow(_Symbol, PERIOD_M30, 0);
    
    double bodySize = (atr > 0) ? MathAbs(close_now - open_0) / atr : 0;
    double upperWick = (atr > 0) ? (high_0 - MathMax(close_now, open_0)) / atr : 0;
    double lowerWick = (atr > 0) ? (MathMin(close_now, open_0) - low_0) / atr : 0;
    int isBullish = (close_now > open_0) ? 1 : 0;
    
    //--- Bars since fractal (simplified: count bars since last fractal)
    int barsFracUp = 999, barsFracDown = 999;
    int handleFrac = iFractals(_Symbol, PERIOD_M30);
    if(handleFrac != INVALID_HANDLE)
    {
        double fUp[], fDown[];
        ArraySetAsSeries(fUp, true);
        ArraySetAsSeries(fDown, true);
        CopyBuffer(handleFrac, 0, 0, 200, fUp);
        CopyBuffer(handleFrac, 1, 0, 200, fDown);
        
        for(int i = 0; i < ArraySize(fUp); i++)
        {
            if(fUp[i] != EMPTY_VALUE && fUp[i] > 0) { barsFracUp = i; break; }
        }
        for(int i = 0; i < ArraySize(fDown); i++)
        {
            if(fDown[i] != EMPTY_VALUE && fDown[i] > 0) { barsFracDown = i; break; }
        }
        IndicatorRelease(handleFrac);
    }
    
    //--- Volume ratio (if available)
    double volRatio = 0;
    long vol_now = iVolume(_Symbol, PERIOD_M30, 0);
    double volSum = 0;
    for(int i = 1; i <= 20; i++)
        volSum += (double)iVolume(_Symbol, PERIOD_M30, i);
    double volAvg = volSum / 20.0;
    volRatio = (volAvg > 0) ? vol_now / volAvg : 0;
    
    //--- Fill feature array (MUST match training order exactly)
    // Order: rsi, atr, daily_range, daily_direction,
    //        dist_res_norm, dist_sup_norm, sr_position,
    //        hour, dow, ret_3, ret_5, ret_10, ret_20,
    //        atr_pctile, atr_change, rsi_sma, rsi_slope,
    //        near_support, near_resistance,
    //        body_size, upper_wick, lower_wick, is_bullish,
    //        bars_since_frac_up, bars_since_frac_down, vol_ratio
    
    features[0]  = (float)rsi;
    features[1]  = (float)atr;
    features[2]  = (float)dailyRange;
    features[3]  = (float)dailyDir;
    features[4]  = (float)distResNorm;
    features[5]  = (float)distSupNorm;
    features[6]  = (float)srPosition;
    features[7]  = (float)hour;
    features[8]  = (float)dow;
    features[9]  = (float)ret3;
    features[10] = (float)ret5;
    features[11] = (float)ret10;
    features[12] = (float)ret20;
    features[13] = (float)atrPctile;
    features[14] = (float)atrChange;
    features[15] = (float)rsiSma;
    features[16] = (float)rsiSlope;
    features[17] = (float)nearSupport;
    features[18] = (float)nearResistance;
    features[19] = (float)bodySize;
    features[20] = (float)upperWick;
    features[21] = (float)lowerWick;
    features[22] = (float)isBullish;
    features[23] = (float)barsFracUp;
    features[24] = (float)barsFracDown;
    features[25] = (float)volRatio;
    
    return true;
}

//+------------------------------------------------------------------+
//| PredictConfidence — ONNX inference                                |
//+------------------------------------------------------------------+
double PredictConfidence(float &features[])
{
    if(!g_onnxLoaded || g_onnxHandle == INVALID_HANDLE)
        return 1.0; // If NN not loaded, pass all signals
    
    //--- Prepare input
    float inputData[];
    ArrayResize(inputData, 26);
    ArrayCopy(inputData, features);
    
    //--- Run inference
    // Output 0: predicted class (int64)
    // Output 1: probabilities (float, shape [1, 2])
    long   predictedClass[];
    float  probabilities[];
    ArrayResize(predictedClass, 1);
    ArrayResize(probabilities, 2);
    
    if(!OnnxRun(g_onnxHandle, 
                ONNX_NO_CONVERSION,
                inputData,
                predictedClass,
                probabilities))
    {
        Print("WARNING: ONNX inference failed. Error: ", GetLastError());
        return 0.5; // uncertain
    }
    
    //--- Return probability of class 1 (GOOD entry)
    double confidence = (double)probabilities[1];
    
    return confidence;
}

//+------------------------------------------------------------------+
//| ExecuteBuy                                                         |
//+------------------------------------------------------------------+
bool ExecuteBuy(double atrPoints)
{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl = 0, tp = 0;
    
    if(InpUseStopLoss)
    {
        if(InpUseATR_SLTP)
            sl = ask - atrPoints * InpSL_ATR_Mult * g_point;
        else
            sl = ask - InpFixedSL_Points * g_point;
    }
    
    if(InpUseTakeProfit)
    {
        if(InpUseATR_SLTP)
            tp = ask + atrPoints * InpTP_ATR_Mult * g_point;
        else
            tp = ask + InpFixedTP_Points * g_point;
    }
    
    sl = NormalizeDouble(sl, g_digits);
    tp = NormalizeDouble(tp, g_digits);
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action    = TRADE_ACTION_DEAL;
    request.symbol    = _Symbol;
    request.volume    = InpLotSize;
    request.type      = ORDER_TYPE_BUY;
    request.price     = ask;
    request.sl        = sl;
    request.tp        = tp;
    request.magic     = InpMagicNumber;
    request.comment   = InpComment + "_BUY";
    request.deviation = 30;
    
    if(!OrderSend(request, result))
    {
        Print("BUY failed: ", result.retcode, " - ", result.comment);
        return false;
    }
    
    g_lastTradeTime = (double)TimeCurrent();
    g_entryATR[0] = atrPoints;
    Print("BUY executed at ", ask, " SL=", sl, " TP=", tp, " ATR=", atrPoints);
    return true;
}

//+------------------------------------------------------------------+
//| ExecuteSell                                                        |
//+------------------------------------------------------------------+
bool ExecuteSell(double atrPoints)
{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double sl = 0, tp = 0;
    
    if(InpUseStopLoss)
    {
        if(InpUseATR_SLTP)
            sl = bid + atrPoints * InpSL_ATR_Mult * g_point;
        else
            sl = bid + InpFixedSL_Points * g_point;
    }
    
    if(InpUseTakeProfit)
    {
        if(InpUseATR_SLTP)
            tp = bid - atrPoints * InpTP_ATR_Mult * g_point;
        else
            tp = bid - InpFixedTP_Points * g_point;
    }
    
    sl = NormalizeDouble(sl, g_digits);
    tp = NormalizeDouble(tp, g_digits);
    
    MqlTradeRequest request = {};
    MqlTradeResult result = {};
    
    request.action    = TRADE_ACTION_DEAL;
    request.symbol    = _Symbol;
    request.volume    = InpLotSize;
    request.type      = ORDER_TYPE_SELL;
    request.price     = bid;
    request.sl        = sl;
    request.tp        = tp;
    request.magic     = InpMagicNumber;
    request.comment   = InpComment + "_SELL";
    request.deviation = 30;
    
    if(!OrderSend(request, result))
    {
        Print("SELL failed: ", result.retcode, " - ", result.comment);
        return false;
    }
    
    g_lastTradeTime = (double)TimeCurrent();
    g_entryATR[0] = atrPoints;
    Print("SELL executed at ", bid, " SL=", sl, " TP=", tp, " ATR=", atrPoints);
    return true;
}

//+------------------------------------------------------------------+
//| ManageSmartTrailing                                               |
//+------------------------------------------------------------------+
void ManageSmartTrailing()
{
    if(!InpUseSmartTrailing) return;
    
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
        
        double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double currentSL = PositionGetDouble(POSITION_SL);
        long posType = PositionGetInteger(POSITION_TYPE);
        
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        
        // Use entry ATR for trailing trigger
        double entryATR = g_entryATR[0];
        if(entryATR <= 0) entryATR = GetATR_M1_Points();
        
        double triggerDist = entryATR * InpTrail_ATR_Mult * g_point;
        double trailStep = InpTrailStepPoints * g_point;
        double minMove = InpMinSLMovement * g_point;
        
        if(posType == POSITION_TYPE_BUY)
        {
            double profit = bid - openPrice;
            if(profit > triggerDist)
            {
                double newSL = bid - trailStep;
                newSL = NormalizeDouble(newSL, g_digits);
                
                if(newSL > currentSL + minMove && newSL > openPrice)
                {
                    MqlTradeRequest request = {};
                    MqlTradeResult result = {};
                    request.action   = TRADE_ACTION_SLTP;
                    request.position = ticket;
                    request.symbol   = _Symbol;
                    request.sl       = newSL;
                    request.tp       = PositionGetDouble(POSITION_TP);
                    
                    if(OrderSend(request, result))
                        Print("BUY Trail: SL moved to ", newSL);
                }
            }
        }
        else if(posType == POSITION_TYPE_SELL)
        {
            double profit = openPrice - ask;
            if(profit > triggerDist)
            {
                double newSL = ask + trailStep;
                newSL = NormalizeDouble(newSL, g_digits);
                
                if((currentSL == 0 || newSL < currentSL - minMove) && newSL < openPrice)
                {
                    MqlTradeRequest request = {};
                    MqlTradeResult result = {};
                    request.action   = TRADE_ACTION_SLTP;
                    request.position = ticket;
                    request.symbol   = _Symbol;
                    request.sl       = newSL;
                    request.tp       = PositionGetDouble(POSITION_TP);
                    
                    if(OrderSend(request, result))
                        Print("SELL Trail: SL moved to ", newSL);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    //--- Manage trailing for existing positions
    ManageSmartTrailing();
    
    //--- Check if we can open new trades
    if(!IsWithinTradingHours())
        return;
    
    if(CountOpenPositions() >= InpMaxOpenTrades)
        return;
    
    if(!IsMinBarsElapsed())
        return;
    
    if(!IsSpreadAcceptable())
        return;
    
    //--- Get indicator values
    double rsiValue = GetRSI_M1();
    double atrPoints = GetATR_M1_Points();
    
    //--- ATR filter
    if(atrPoints < InpATR_MinPoints || atrPoints > InpATR_MaxPoints)
        return;
    
    //--- Daily direction
    int dailyDirection = DetectDailyRange();
    if(dailyDirection == 0)
        return;
    
    //--- Build S/R levels
    BuildSNR();
    
    //--- Get prices
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double tolerance = InpSNR_Tolerance * g_point;
    
    //--- Check BUY conditions
    bool buySignal = false;
    if(dailyDirection == 1 && g_currentSupport > 0)
    {
        if(ask >= (g_currentSupport - tolerance) && ask <= (g_currentSupport + tolerance))
        {
            if(rsiValue <= InpRSI_Oversold)
                buySignal = true;
        }
    }
    
    //--- Check SELL conditions
    bool sellSignal = false;
    if(dailyDirection == -1 && g_currentResistance > 0)
    {
        if(bid <= (g_currentResistance + tolerance) && bid >= (g_currentResistance - tolerance))
        {
            if(rsiValue >= InpRSI_Overbought)
                sellSignal = true;
        }
    }
    
    //--- If no signal, return
    if(!buySignal && !sellSignal)
        return;
    
    //--- NN FILTER — THE KEY ADDITION
    if(InpUseNNFilter)
    {
        float nnFeatures[];
        if(!PrepareFeatures(nnFeatures))
        {
            Print("WARNING: Could not prepare features for NN");
            return;
        }
        
        double confidence = PredictConfidence(nnFeatures);
        
        if(confidence < InpConfidenceThresh)
        {
            string sigType = buySignal ? "BUY" : "SELL";
            Print("NN FILTER: ", sigType, " signal REJECTED | Confidence: ", 
                  DoubleToString(confidence, 4), " < ", 
                  DoubleToString(InpConfidenceThresh, 2),
                  " | RSI=", DoubleToString(rsiValue, 1),
                  " | ATR=", DoubleToString(atrPoints, 0));
            return;
        }
        
        string sigType = buySignal ? "BUY" : "SELL";
        Print("NN FILTER: ", sigType, " signal ACCEPTED | Confidence: ", 
              DoubleToString(confidence, 4), " >= ", 
              DoubleToString(InpConfidenceThresh, 2));
    }
    
    //--- Execute trade
    if(buySignal)
    {
        ExecuteBuy(atrPoints);
    }
    else if(sellSignal)
    {
        ExecuteSell(atrPoints);
    }
}
//+------------------------------------------------------------------+
