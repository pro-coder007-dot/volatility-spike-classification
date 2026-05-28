import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="NVIDIA Volatility Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NVIDIA Stock Volatility Spike Predictor")
st.markdown("Predict whether next-day volatility will exceed the 90th percentile threshold")

tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "CSV Upload"])

with tab1:
    st.subheader("Enter Stock Data Manually")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        adj_close = st.number_input("Adj Close", value=6.08, format="%.6f")
        high = st.number_input("High", value=6.23, format="%.6f")
        low = st.number_input("Low", value=6.11, format="%.6f")
        open_price = st.number_input("Open", value=6.14, format="%.6f")
    
    with col2:
        close = st.number_input("Close", value=6.14, format="%.6f")
        volume = st.number_input("Volume", value=478576000.0, format="%.0f")
        return_val = st.number_input("Return", value=0.0127, format="%.6f")
        volatility = st.number_input("Volatility", value=0.0190, format="%.6f")
    
    with col3:
        day_of_week = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], 
                                    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        month = st.slider("Month", 1, 12, 1)
        day = st.slider("Day", 1, 31, 31)
        week_of_year = st.slider("Week of Year", 1, 53, 5)
    
    if st.button("🔮 Predict", key="single_predict"):
        with st.spinner("Running prediction..."):
            payload = {
                "adj_close": adj_close,
                "close": close,
                "high": high,
                "low": low,
                "open": open_price,
                "volume": volume,
                "day_of_week": day_of_week,
                "month": month,
                "day": day,
                "week_of_year": week_of_year,
                "return_": return_val,
                "volatility": volatility
            }
            
            try:
                response = requests.post(f"{API_URL}/predict", json=payload)
                response.raise_for_status()
                result = response.json()
                
                st.markdown("---")
                
                if result["spike_predicted"]:
                    st.error(f"⚠️ **VOLATILITY SPIKE PREDICTED**")
                    st.metric("Spike Probability", f"{result['probability']:.2%}")
                else:
                    st.success(f"✅ **Normal Volatility Expected**")
                    st.metric("Spike Probability", f"{result['probability']:.2%}")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ API Error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with tab2:
    st.subheader("Predict Multiple Entries")
    st.info("Add multiple rows of data and predict all at once")
    
    if 'batch_data' not in st.session_state:
        st.session_state.batch_data = []
    
    with st.form("batch_form"):
        st.write("Add a row:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            b_adj_close = st.number_input("Adj Close", value=6.08, key="b_adj")
            b_high = st.number_input("High", value=6.23, key="b_high")
            b_low = st.number_input("Low", value=6.11, key="b_low")
        
        with col2:
            b_open = st.number_input("Open", value=6.14, key="b_open")
            b_close = st.number_input("Close", value=6.14, key="b_close")
            b_volume = st.number_input("Volume", value=478576000.0, key="b_vol")
        
        with col3:
            b_return = st.number_input("Return", value=0.0127, key="b_ret")
            b_volatility = st.number_input("Volatility", value=0.0190, key="b_volat")
            b_dow = st.selectbox("Day of Week", [0,1,2,3,4,5,6], key="b_dow")
        
        with col4:
            b_month = st.slider("Month", 1, 12, 1, key="b_month")
            b_day = st.slider("Day", 1, 31, 31, key="b_day")
            b_woy = st.slider("Week of Year", 1, 53, 5, key="b_woy")
        
        submitted = st.form_submit_button("➕ Add to Batch")
        
        if submitted:
            row = {
                "adj_close": b_adj_close,
                "close": b_close,
                "high": b_high,
                "low": b_low,
                "open": b_open,
                "volume": b_volume,
                "day_of_week": b_dow,
                "month": b_month,
                "day": b_day,
                "week_of_year": b_woy,
                "return_": b_return,
                "volatility": b_volatility
            }
            st.session_state.batch_data.append(row)
            st.success(f"✅ Added row {len(st.session_state.batch_data)}")
    
    if st.session_state.batch_data:
        st.write(f"**Current batch: {len(st.session_state.batch_data)} rows**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔮 Predict Batch"):
                with st.spinner("Running batch prediction..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            json=st.session_state.batch_data
                        )
                        response.raise_for_status()
                        results = response.json()
                        
                        st.markdown("---")
                        st.subheader("Batch Results")
                        
                        df_results = pd.DataFrame(results)
                        
                        spike_count = df_results['spike_predicted'].sum()
                        total = len(df_results)
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Total Predictions", total)
                        col_b.metric("Spikes Detected", spike_count)
                        col_c.metric("Spike Rate", f"{(spike_count/total)*100:.2f}%")
                        
                        st.write("**Probability Distribution:**")
                        col_x, col_y, col_z = st.columns(3)
                        col_x.metric("Min Probability", f"{df_results['probability'].min():.4f}")
                        col_y.metric("Mean Probability", f"{df_results['probability'].mean():.4f}")
                        col_z.metric("Max Probability", f"{df_results['probability'].max():.4f}")
                        
                        st.write("**All Results:**")
                        st.dataframe(df_results, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear Batch"):
                st.session_state.batch_data = []
                st.rerun()

with tab3:
    st.subheader("Upload CSV File")
    st.info("Upload a CSV with columns: Adj_Close, High, Low, Open, Close, Volume, day_of_week, month, day, week_of_year, return, volatility")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Preview (first 5 rows):**")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🔮 Predict from CSV"):
            with st.spinner("Processing CSV..."):
                try:
                    uploaded_file.seek(0)
                    
                    files = {"file": ("data.csv", uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/predict/csv", files=files)
                    response.raise_for_status()
                    result = response.json()
                    
                    st.markdown("---")
                    st.subheader("CSV Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Rows", result["total_rows"])
                    col2.metric("Spikes Detected", result["spike_count"])
                    col3.metric("Spike Rate", f"{result['spike_percentage']:.2f}%")
                    
                    df_results = pd.DataFrame(result["predictions"])
                    
                    st.write("**Probability Distribution:**")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Min Probability", f"{df_results['probability'].min():.4f}")
                    col_b.metric("Mean Probability", f"{df_results['probability'].mean():.4f}")
                    col_c.metric("Max Probability", f"{df_results['probability'].max():.4f}")
                    
                    st.write("**Sample Predictions (first 10 rows):**")
                    st.dataframe(df_results.head(10), use_container_width=True)
                    
                    with st.expander("📊 View All Results"):
                        st.dataframe(df_results, use_container_width=True)
                    
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

with st.sidebar:
    st.header("⚙️ System Status")
    
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                st.success("✅ API is running")
                data = response.json()
                st.json(data)
            else:
                st.error("❌ API is down")
        except:
            st.error("❌ Cannot connect to API")
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    if st.button("Get Feature Importance"):
        try:
            response = requests.get(f"{API_URL}/model/feature-importance")
            if response.status_code == 200:
                data = response.json()
                
                df_importance = pd.DataFrame(data["feature_importance"])
                df_importance = df_importance.sort_values("importance", ascending=False)
                
                st.dataframe(df_importance, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Built with FastAPI + Streamlit | NVIDIA Volatility Predictor")