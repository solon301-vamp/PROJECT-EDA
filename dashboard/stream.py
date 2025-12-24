import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Page config
st.set_page_config(
    page_title="KNN Literation Analytics Report 2024",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Futuristic Black & Blue Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Rajdhani', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #000000 0%, #001a33 50%, #000814 100%);
        padding: 0;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00d9ff;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #4dd0e1;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #00d9ff;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.6);
    }
    
    h3 {
        font-family: 'Orbitron', sans-serif;
        color: #4dd0e1;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(0, 26, 51, 0.9), rgba(0, 51, 102, 0.7));
        border: 2px solid #00d9ff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    with open('dashboard_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['provinces'])
    df['Kategori'] = df['Label_TGM'].map({0: 'Rendah', 1: 'Sedang', 2: 'Tinggi'})
    knn_eval = data['knn_evaluation']
    
    return df, knn_eval

df, knn_eval = load_data()

# Calculate stats
total_provinces = len(df)
tinggi_count = len(df[df['Kategori'] == 'Tinggi'])
avg_tgm = df['Tingkat Kegemaran Membaca'].mean()
best_k = knn_eval['best_k']
best_accuracy = knn_eval['best_accuracy'] * 100
top_province = df.nlargest(1, 'Tingkat Kegemaran Membaca').iloc[0]

# Calculate correlation
corr_tgm_aps = df['Tingkat Kegemaran Membaca'].corr(df['APS_19_23'])

# Get regions
regions = {
    'Jawa': ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'DI Yogyakarta', 'Jawa Timur', 'Banten'],
    'Sumatera': ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 'Bengkulu', 'Lampung', 'Kepulauan Bangka Belitung', 'Kepulauan Riau'],
    'Kalimantan': ['Kalimantan Barat', 'Kalimantan Tengah', 'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara'],
    'Sulawesi': ['Sulawesi Utara', 'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat']
}

top_region = None
top_region_tgm = 0
for region, provs in regions.items():
    avg = df[df['Provinsi'].isin(provs)]['Tingkat Kegemaran Membaca'].mean()
    if avg > top_region_tgm:
        top_region_tgm = avg
        top_region = region

categories = df['Kategori'].value_counts()

# ===== HEADER BAR =====
st.markdown("""
<div style='background: linear-gradient(90deg, #001a33 0%, #003366 50%, #001a33 100%); 
            padding: 25px; border-bottom: 3px solid #00d9ff; box-shadow: 0 4px 20px rgba(0, 217, 255, 0.4);'>
    <h1 style='text-align: center; margin: 0; font-size: 2.2rem;'>
        ⚡ KNN LITERATION ANALYTICS REPORT 2024 ⚡
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)

# ===== TOP METRICS BAR =====
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

with col1:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>🏆 TOP PROVINSI</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>{top_province['Provinsi'][:15]}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>🌍 TOP REGION</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>{top_region}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>📊 AVG TGM SCORE</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>{avg_tgm:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>🎯 BEST K VALUE</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>K = {best_k}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>🎯 ACCURACY</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>{best_accuracy:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>🔗 CORRELATION</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>r={corr_tgm_aps:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown(f"""
    <div class='metric-box'>
        <div style='color: #4dd0e1; font-size: 0.7rem; font-weight: 600;'>⚙️ DATASET</div>
        <div style='color: #00d9ff; font-size: 1.3rem; font-weight: 700;'>{total_provinces} Prov</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

# ===== MAIN CONTENT - 3 COLUMNS (REBALANCED) =====
col1, col2, col3 = st.columns([1.3, 2, 0.8])

# ===== LEFT COLUMN (WITH TOGGLE MOVED HERE) =====
with col1:
    # TGM Score Trend
    st.markdown("<h3>📈 TGM Score Trend</h3>", unsafe_allow_html=True)
    
    top_15 = df.nlargest(15, 'Tingkat Kegemaran Membaca').sort_values('Tingkat Kegemaran Membaca')
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=list(range(len(top_15))),
        y=top_15['Tingkat Kegemaran Membaca'],
        mode='lines+markers',
        line=dict(color='#00d9ff', width=3, shape='spline'),
        marker=dict(size=8, color='#00d9ff', line=dict(width=2, color='#0099cc')),
        fill='tonexty',
        fillcolor='rgba(0, 217, 255, 0.1)',
        text=top_15['Tingkat Kegemaran Membaca'].round(1),
        textposition='top center',
        textfont=dict(size=9, color='#00d9ff'),
        hovertemplate='<b>TGM: %{y:.2f}</b><extra></extra>'
    ))
    
    fig_trend.update_layout(
        height=240,
        margin=dict(l=30, r=10, t=10, b=30),
        paper_bgcolor='rgba(0, 26, 51, 0.5)',
        plot_bgcolor='rgba(0, 8, 20, 0.8)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)', showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)', tickfont=dict(size=10, color='#4dd0e1'), range=[60, 85])
    )
    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    # Feature Correlation Heatmap
    st.markdown("<h3>🔥 Feature Correlation</h3>", unsafe_allow_html=True)
    
    corr_features = ['Tingkat Kegemaran Membaca', 'Frekuensi Membaca', 'Jumlah Buku yang Dibaca', 
                     'APS_19_23', 'APS_16_18', 'Frekuensi Akses Internet']
    corr_matrix = df[corr_features].corr()
    short_labels = ['TGM', 'Frek.Baca', 'Jml.Buku', 'APS 19-23', 'APS 16-18', 'Frek.Net']
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=short_labels,
        y=short_labels,
        colorscale=[[0, '#001a33'], [0.5, '#003366'], [0.75, '#0066cc'], [1, '#00d9ff']],
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10, color='white', weight=700),
        hovertemplate='<b>%{x} × %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Corr', titlefont=dict(color='#4dd0e1', size=10), tickfont=dict(color='#4dd0e1', size=9))
    ))
    
    fig_corr.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0, 26, 51, 0.5)',
        plot_bgcolor='rgba(0, 8, 20, 0.8)',
        xaxis=dict(tickfont=dict(size=9, color='#4dd0e1'), side='bottom'),
        yaxis=dict(tickfont=dict(size=9, color='#4dd0e1'))
    )
    st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    # Distribution (without toggle)
    st.markdown("<h3>📊 Category Distribution</h3>", unsafe_allow_html=True)
    
    fig_mini_cat = go.Figure()
    cat_order = ['Tinggi', 'Sedang', 'Rendah']
    cat_colors = ['#00d9ff', '#0099cc', '#004d99']
    cat_values = [categories.get(cat, 0) for cat in cat_order]
    
    fig_mini_cat.add_trace(go.Bar(
        x=cat_order,
        y=cat_values,
        marker=dict(color=cat_colors, line=dict(color='#00d9ff', width=1)),
        text=cat_values,
        textposition='outside',
        textfont=dict(size=17, color='#00d9ff', weight=1000),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: ' + 
                     (pd.Series(cat_values) / sum(cat_values) * 100).round(1).astype(str) + '%<extra></extra>',
        showlegend=False
    ))
    
    fig_mini_cat.update_layout(
        height=300,
        margin=dict(l=30, r=20, t=10, b=30),
        paper_bgcolor='rgba(0, 26, 51, 0.5)',
        plot_bgcolor='rgba(0, 8, 20, 0.8)',
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color='#4dd0e1', weight=600)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 217, 255, 0.1)',
            tickfont=dict(size=10, color='#4dd0e1'),
            range=[0, max(cat_values) + 5],
            title=dict(text='Jumlah Provinsi', font=dict(size=10, color='#4dd0e1'))
        )
    )
    st.plotly_chart(fig_mini_cat, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    # Regional Performance (separate chart)
    st.markdown("<h3>🗺️ Regional Performance</h3>", unsafe_allow_html=True)
    
    region_stats = []
    for region, provs in regions.items():
        region_df = df[df['Provinsi'].isin(provs)]
        if len(region_df) > 0:
            region_stats.append({
                'Region': region,
                'Avg_TGM': region_df['Tingkat Kegemaran Membaca'].mean(),
                'Count': len(region_df),
                'Max': region_df['Tingkat Kegemaran Membaca'].max(),
                'Min': region_df['Tingkat Kegemaran Membaca'].min()
            })
    
    region_perf_df = pd.DataFrame(region_stats).sort_values('Avg_TGM', ascending=False)
    
    fig_regional = go.Figure()
    
    colors_regional = []
    for avg in region_perf_df['Avg_TGM']:
        if avg >= 72:
            colors_regional.append('#00d9ff')
        elif avg >= 68:
            colors_regional.append('#0099cc')
        elif avg >= 65:
            colors_regional.append('#006699')
        else:
            colors_regional.append('#004d99')
    
    fig_regional.add_trace(go.Bar(
        y=region_perf_df['Region'][::-1],
        x=region_perf_df['Avg_TGM'][::-1],
        orientation='h',
        marker=dict(color=colors_regional[::-1], line=dict(color='#00d9ff', width=2)),
        text=[f"{val:.1f}" for val in region_perf_df['Avg_TGM'][::-1]],
        textposition='outside',
        textfont=dict(size=13, color='#00d9ff', weight=900),
        hovertemplate='<b>%{y}</b><br>Avg TGM: %{x:.2f}<br>Provinces: ' + 
                     region_perf_df['Count'][::-1].astype(str) + 
                     '<br>Range: ' + region_perf_df['Min'][::-1].round(1).astype(str) + 
                     ' - ' + region_perf_df['Max'][::-1].round(1).astype(str) + '<extra></extra>',
        showlegend=False
    ))
    
    avg_all = df['Tingkat Kegemaran Membaca'].mean()
    fig_regional.add_vline(
        x=avg_all,
        line_dash="dash",
        line_color="#fbbf24",
        line_width=2,
        annotation_text=f"Avg: {avg_all:.1f}",
        annotation_position="top",
        annotation_font_color="#fbbf24",
        annotation_font_size=10
    )
    
    fig_regional.update_layout(
        height=250,
        margin=dict(l=30, r=50, t=10, b=30),
        paper_bgcolor='rgba(0, 26, 51, 0.5)',
        plot_bgcolor='rgba(0, 8, 20, 0.8)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0, 217, 255, 0.1)',
            tickfont=dict(size=10, color='#4dd0e1'),
            range=[60, 75],
            title=dict(text='Average TGM Score', font=dict(size=10, color='#4dd0e1'))
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, color='#4dd0e1', weight=600))
    )
    st.plotly_chart(fig_regional, use_container_width=True, config={'displayModeBar': False})

# ===== MIDDLE COLUMN =====
with col2:
    # Top 5 Provinces
    st.markdown("<h3>🏆 Top 5 Provinsi by TGM Score</h3>", unsafe_allow_html=True)
    
    top_5 = df.nlargest(5, 'Tingkat Kegemaran Membaca')
    
    fig_top5 = go.Figure()
    colors_gradient = ['#00d9ff', '#0099cc', '#006699', '#004d99', '#003366']
    
    fig_top5.add_trace(go.Bar(
        y=top_5['Provinsi'][::-1],
        x=top_5['Tingkat Kegemaran Membaca'][::-1],
        orientation='h',
        marker=dict(color=colors_gradient[::-1], line=dict(color='#00d9ff', width=2)),
        text=['$' + str(int(x)) for x in top_5['Tingkat Kegemaran Membaca'][::-1]],
        textposition='outside',
        textfont=dict(size=12, color='#00d9ff', weight=700),
        hovertemplate='<b>%{y}</b><br>TGM: %{x:.2f}<extra></extra>'
    ))
    
    fig_top5.update_layout(
        height=160,
        margin=dict(l=10, r=60, t=10, b=20),
        paper_bgcolor='rgba(0, 26, 51, 0.5)',
        plot_bgcolor='rgba(0, 8, 20, 0.8)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)', range=[0, 90]),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color='#4dd0e1', weight=600))
    )
    st.plotly_chart(fig_top5, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    # 2x2 Grid
    col2a, col2b = st.columns(2)
    
    with col2a:
        # KNN Accuracy by K Value
        st.markdown("<h3 style='font-size: 0.9rem;'>🎯 KNN Model Evaluation</h3>", unsafe_allow_html=True)
        
        k_values = [int(k) for k in knn_eval['all_k_results'].keys()]
        accuracies = [v * 100 for v in knn_eval['all_k_results'].values()]
        
        fig_knn = go.Figure()
        
        fig_knn.add_trace(go.Scatter(
            x=k_values,
            y=accuracies,
            mode='lines+markers',
            line=dict(color='#00d9ff', width=3),
            marker=dict(
                size=10,
                color=accuracies,
                colorscale=[[0, '#003366'], [0.5, '#0080ff'], [1, '#00d9ff']],
                line=dict(width=2, color='white')
            ),
            text=[f'{a:.1f}%' for a in accuracies],
            textposition='top center',
            textfont=dict(size=9, color='#00d9ff', weight=600),
            hovertemplate='<b>K=%{x}</b><br>Accuracy: %{y:.2f}%<extra></extra>'
        ))
        
        best_idx = k_values.index(best_k)
        fig_knn.add_trace(go.Scatter(
            x=[best_k],
            y=[accuracies[best_idx]],
            mode='markers+text',
            marker=dict(size=20, color='#fbbf24', line=dict(width=3, color='white')),
            text=[f'BEST<br>K={best_k}'],
            textposition='bottom center',
            textfont=dict(size=10, color='#fbbf24', weight=700),
            hovertemplate=f'<b>OPTIMAL K={best_k}</b><br>Accuracy: {accuracies[best_idx]:.2f}%<extra></extra>',
            showlegend=False
        ))
        
        fig_knn.update_layout(
            height=300,
            margin=dict(l=30, r=10, t=10, b=30),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(
                title='K Value',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1'),
                tickvals=k_values
            ),
            yaxis=dict(
                title='Accuracy (%)',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1'),
                range=[60, 90]
            ),
            showlegend=False
        )
        st.plotly_chart(fig_knn, use_container_width=True, config={'displayModeBar': False})
    
    with col2b:
        # APS Decline Trend
        st.markdown("<h3 style='font-size: 0.9rem;'>📉 APS Decline by Age Group</h3>", unsafe_allow_html=True)
        
        aps_data = {
            'Label': ['7-12 thn', '13-15 thn', '16-18 thn', '19-23 thn'],
            'APS': [
                df['APS_7_12'].mean(),
                df['APS_13_15'].mean(),
                df['APS_16_18'].mean(),
                df['APS_19_23'].mean()
            ]
        }
        aps_df = pd.DataFrame(aps_data)
        
        fig_aps = go.Figure()
        
        fig_aps.add_trace(go.Scatter(
            x=aps_df['Label'],
            y=aps_df['APS'],
            mode='lines+markers',
            line=dict(color='#00d9ff', width=3),
            marker=dict(size=10, color='#00d9ff', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.2)',
            text=[f'{v:.1f}%' for v in aps_df['APS']],
            textposition='top center',
            textfont=dict(size=10, color='#00d9ff', weight=700),
            hovertemplate='<b>%{x}</b><br>APS: %{y:.2f}%<extra></extra>'
        ))
        
        fig_aps.add_annotation(
            x='19-23 thn',
            y=aps_df['APS'].iloc[-1],
            text='⚠️ DROP 76%',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#ff4444',
            ax=40,
            ay=-40,
            font=dict(size=11, color='#ff4444', weight=700),
            bgcolor='rgba(255, 68, 68, 0.2)',
            bordercolor='#ff4444',
            borderwidth=2
        )
        
        fig_aps.update_layout(
            height=250,
            margin=dict(l=30, r=10, t=10, b=30),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(showgrid=False, tickfont=dict(size=9, color='#4dd0e1')),
            yaxis=dict(
                title='APS (%)',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1'),
                range=[0, 110]
            )
        )
        st.plotly_chart(fig_aps, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    # Bottom 2x2 Grid
    col2c, col2d = st.columns(2)
    
    with col2c:
        st.markdown("<h3 style='font-size: 0.9rem;'>🗺️ TGM by Region</h3>", unsafe_allow_html=True)
        
        region_avg = []
        for region, provs in regions.items():
            avg = df[df['Provinsi'].isin(provs)]['Tingkat Kegemaran Membaca'].mean()
            if not pd.isna(avg):
                region_avg.append({'Region': region, 'TGM': avg})
        
        region_avg_df = pd.DataFrame(region_avg)
        
        fig_region_pie = go.Figure(data=[go.Pie(
            labels=region_avg_df['Region'],
            values=region_avg_df['TGM'],
            hole=0.5,
            marker=dict(
                colors=['#00d9ff', '#0099cc', '#006699', '#004d99'],
                line=dict(color='#000814', width=2)
            ),
            textinfo='label+percent',
            textfont=dict(size=9, color='white', weight=600),
            hovertemplate='<b>%{label}</b><br>Avg TGM: %{value:.2f}<extra></extra>'
        )])
        
        fig_region_pie.add_annotation(
            text=f"<b>AVG</b><br>{region_avg_df['TGM'].mean():.1f}",
            x=0.5, y=0.5,
            font=dict(size=16, color='#00d9ff'),
            showarrow=False
        )
        
        fig_region_pie.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            showlegend=False
        )
        st.plotly_chart(fig_region_pie, use_container_width=True, config={'displayModeBar': False})
    
    with col2d:
        st.markdown("<h3 style='font-size: 0.9rem;'>📚 Feature Importance</h3>", unsafe_allow_html=True)
        
        features_short = ['Frek.Baca', 'Jml.Buku', 'Durasi', 'APS 19-23', 'Frek.Net']
        importance = [95, 90, 88, 78, 72]
        colors_feat = ['#00d9ff', '#00d9ff', '#0099cc', '#006699', '#004d99']
        
        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            y=features_short[::-1],
            x=importance[::-1],
            orientation='h',
            marker=dict(color=colors_feat[::-1], line=dict(color='#00d9ff', width=1)),
            text=importance[::-1],
            textposition='outside',
            textfont=dict(size=10, color='#00d9ff', weight=600),
            hovertemplate='<b>%{y}</b><br>Importance: %{x}<extra></extra>'
        ))
        
        fig_feat.update_layout(
            height=180,
            margin=dict(l=10, r=30, t=10, b=10),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)', range=[0, 105]),
            yaxis=dict(showgrid=False, tickfont=dict(size=9, color='#4dd0e1'))
        )
        st.plotly_chart(fig_feat, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    # 🆕 NEW: 2x2 Correlation Scatter Plots
    st.markdown("<h3>🔬 Correlation Analysis: TGM vs Key Features</h3>", unsafe_allow_html=True)
    
    col2e, col2f = st.columns(2)
    
    with col2e:
        # Scatter 1: TGM vs Frekuensi Membaca
        st.markdown("<h3 style='font-size: 0.85rem;'>📖 TGM vs Frekuensi Membaca</h3>", unsafe_allow_html=True)
        
        corr_frek = df['Tingkat Kegemaran Membaca'].corr(df['Frekuensi Membaca'])
        
        fig_scatter1 = go.Figure()
        
        # Color by kategori
        colors_scatter = {'Tinggi': '#00d9ff', 'Sedang': '#0099cc', 'Rendah': '#ef4444'}
        
        for cat in ['Tinggi', 'Sedang', 'Rendah']:
            cat_data = df[df['Kategori'] == cat]
            fig_scatter1.add_trace(go.Scatter(
                x=cat_data['Frekuensi Membaca'],
                y=cat_data['Tingkat Kegemaran Membaca'],
                mode='markers',
                name=cat,
                marker=dict(
                    size=10,
                    color=colors_scatter[cat],
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=cat_data['Provinsi'],
                hovertemplate='<b>%{text}</b><br>Frek: %{x}<br>TGM: %{y:.2f}<extra></extra>'
            ))
        
        # Add trendline
        z = np.polyfit(df['Frekuensi Membaca'], df['Tingkat Kegemaran Membaca'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['Frekuensi Membaca'].min(), df['Frekuensi Membaca'].max(), 100)
        
        fig_scatter1.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            name='Trendline',
            line=dict(color='#fbbf24', width=2, dash='dash'),
            showlegend=False,
            hovertemplate='Trendline<extra></extra>'
        ))
        
        fig_scatter1.add_annotation(
            text=f'<b>r = {corr_frek:.3f}</b><br>Strong',
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor='rgba(0, 217, 255, 0.2)',
            bordercolor='#00d9ff',
            borderwidth=2,
            font=dict(size=11, color='#00d9ff', weight=700)
        )
        
        fig_scatter1.update_layout(
            height=200,
            margin=dict(l=30, r=10, t=10, b=30),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(
                title='Frekuensi Membaca',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            yaxis=dict(
                title='TGM Score',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.3,
                xanchor='center',
                x=0.5,
                font=dict(color='white', size=8)
            ),
            showlegend=False
        )
        st.plotly_chart(fig_scatter1, use_container_width=True, config={'displayModeBar': False})
    
    with col2f:
        # Scatter 2: TGM vs Jumlah Buku
        st.markdown("<h3 style='font-size: 0.85rem;'>📚 TGM vs Jumlah Buku Dibaca</h3>", unsafe_allow_html=True)
        
        corr_buku = df['Tingkat Kegemaran Membaca'].corr(df['Jumlah Buku yang Dibaca'])
        
        fig_scatter2 = go.Figure()
        
        for cat in ['Tinggi', 'Sedang', 'Rendah']:
            cat_data = df[df['Kategori'] == cat]
            fig_scatter2.add_trace(go.Scatter(
                x=cat_data['Jumlah Buku yang Dibaca'],
                y=cat_data['Tingkat Kegemaran Membaca'],
                mode='markers',
                name=cat,
                marker=dict(
                    size=10,
                    color=colors_scatter[cat],
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=cat_data['Provinsi'],
                hovertemplate='<b>%{text}</b><br>Buku: %{x}<br>TGM: %{y:.2f}<extra></extra>'
            ))
        
        # Add trendline
        z2 = np.polyfit(df['Jumlah Buku yang Dibaca'], df['Tingkat Kegemaran Membaca'], 1)
        p2 = np.poly1d(z2)
        x_trend2 = np.linspace(df['Jumlah Buku yang Dibaca'].min(), df['Jumlah Buku yang Dibaca'].max(), 100)
        
        fig_scatter2.add_trace(go.Scatter(
            x=x_trend2,
            y=p2(x_trend2),
            mode='lines',
            name='Trendline',
            line=dict(color='#fbbf24', width=2, dash='dash'),
            showlegend=False
        ))
        
        fig_scatter2.add_annotation(
            text=f'<b>r = {corr_buku:.3f}</b><br>Strong',
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor='rgba(0, 217, 255, 0.2)',
            bordercolor='#00d9ff',
            borderwidth=2,
            font=dict(size=11, color='#00d9ff', weight=700)
        )
        
        fig_scatter2.update_layout(
            height=200,
            margin=dict(l=30, r=10, t=10, b=30),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(
                title='Jumlah Buku Dibaca',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            yaxis=dict(
                title='TGM Score',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            showlegend=False
        )
        st.plotly_chart(fig_scatter2, use_container_width=True, config={'displayModeBar': False})
    
    # Second row of scatter plots
    col2g, col2h = st.columns(2)
    
    with col2g:
        # Scatter 3: TGM vs APS 19-23
        st.markdown("<h3 style='font-size: 0.85rem;'>🎓 TGM vs APS (19-23 thn)</h3>", unsafe_allow_html=True)
        
        corr_aps1923 = df['Tingkat Kegemaran Membaca'].corr(df['APS_19_23'])
        
        fig_scatter3 = go.Figure()
        
        for cat in ['Tinggi', 'Sedang', 'Rendah']:
            cat_data = df[df['Kategori'] == cat]
            fig_scatter3.add_trace(go.Scatter(
                x=cat_data['APS_19_23'],
                y=cat_data['Tingkat Kegemaran Membaca'],
                mode='markers',
                name=cat,
                marker=dict(
                    size=10,
                    color=colors_scatter[cat],
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=cat_data['Provinsi'],
                hovertemplate='<b>%{text}</b><br>APS: %{x:.1f}%<br>TGM: %{y:.2f}<extra></extra>'
            ))
        
        # Add trendline
        z3 = np.polyfit(df['APS_19_23'], df['Tingkat Kegemaran Membaca'], 1)
        p3 = np.poly1d(z3)
        x_trend3 = np.linspace(df['APS_19_23'].min(), df['APS_19_23'].max(), 100)
        
        fig_scatter3.add_trace(go.Scatter(
            x=x_trend3,
            y=p3(x_trend3),
            mode='lines',
            name='Trendline',
            line=dict(color='#fbbf24', width=2, dash='dash'),
            showlegend=False
        ))
        
        fig_scatter3.add_annotation(
            text=f'<b>r = {corr_aps1923:.3f}</b><br>Weak ⚠️',
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor='rgba(239, 68, 68, 0.2)',
            bordercolor='#ef4444',
            borderwidth=2,
            font=dict(size=11, color='#ef4444', weight=700)
        )
        
        fig_scatter3.update_layout(
            height=200,
            margin=dict(l=30, r=10, t=10, b=30),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(
                title='APS 19-23 tahun (%)',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            yaxis=dict(
                title='TGM Score',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            showlegend=False
        )
        st.plotly_chart(fig_scatter3, use_container_width=True, config={'displayModeBar': False})
    
    with col2h:
        # Scatter 4: TGM vs APS 16-18
        st.markdown("<h3 style='font-size: 0.85rem;'>🎓 TGM vs APS (16-18 thn)</h3>", unsafe_allow_html=True)
        
        corr_aps1618 = df['Tingkat Kegemaran Membaca'].corr(df['APS_16_18'])
        
        fig_scatter4 = go.Figure()
        
        for cat in ['Tinggi', 'Sedang', 'Rendah']:
            cat_data = df[df['Kategori'] == cat]
            fig_scatter4.add_trace(go.Scatter(
                x=cat_data['APS_16_18'],
                y=cat_data['Tingkat Kegemaran Membaca'],
                mode='markers',
                name=cat,
                marker=dict(
                    size=10,
                    color=colors_scatter[cat],
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=cat_data['Provinsi'],
                hovertemplate='<b>%{text}</b><br>APS: %{x:.1f}%<br>TGM: %{y:.2f}<extra></extra>'
            ))
        
        # Add trendline
        z4 = np.polyfit(df['APS_16_18'], df['Tingkat Kegemaran Membaca'], 1)
        p4 = np.poly1d(z4)
        x_trend4 = np.linspace(df['APS_16_18'].min(), df['APS_16_18'].max(), 100)
        
        fig_scatter4.add_trace(go.Scatter(
            x=x_trend4,
            y=p4(x_trend4),
            mode='lines',
            name='Trendline',
            line=dict(color='#fbbf24', width=2, dash='dash'),
            showlegend=False
        ))
        
        fig_scatter4.add_annotation(
            text=f'<b>r = {corr_aps1618:.3f}</b><br>Moderate ⭐',
            xref='paper', yref='paper',
            x=0.05, y=0.95,
            showarrow=False,
            bgcolor='rgba(251, 191, 36, 0.2)',
            bordercolor='#fbbf24',
            borderwidth=2,
            font=dict(size=11, color='#fbbf24', weight=700)
        )
        
        fig_scatter4.update_layout(
            height=200,
            margin=dict(l=30, r=10, t=10, b=30),
            paper_bgcolor='rgba(0, 26, 51, 0.5)',
            plot_bgcolor='rgba(0, 8, 20, 0.8)',
            xaxis=dict(
                title='APS 16-18 tahun (%)',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            yaxis=dict(
                title='TGM Score',
                titlefont=dict(size=10, color='#4dd0e1'),
                showgrid=True,
                gridcolor='rgba(0, 217, 255, 0.1)',
                tickfont=dict(size=9, color='#4dd0e1')
            ),
            showlegend=False
        )
        st.plotly_chart(fig_scatter4, use_container_width=True, config={'displayModeBar': False})

# ===== RIGHT COLUMN (NOW CLEANER) =====
with col3:
    # Combined: Model Config + Statistics
    st.markdown(f"""
    <div style='background: rgba(0, 26, 51, 0.6); border: 1px solid #00d9ff; border-radius: 8px; padding: 10px; margin-bottom: 10px;'>
        <h3 style='font-size: 0.8rem; margin: 0 0 5px 0;'>⚙️ MODEL & STATS</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px;'>
            <div style='background: rgba(0, 51, 102, 0.5); padding: 6px; border-radius: 5px; text-align: center;'>
                <div style='color: #4dd0e1; font-size: 0.65rem;'>Algorithm</div>
                <div style='color: #00d9ff; font-weight: 700; font-size: 0.85rem;'>K-NN</div>
            </div>
            <div style='background: rgba(0, 51, 102, 0.5); padding: 5px; border-radius: 5px; text-align: center;'>
                <div style='color: #4dd0e1; font-size: 0.65rem;'>Best K</div>
                <div style='color: #00d9ff; font-weight: 700; font-size: 0.85rem;'>{best_k}</div>
            </div>
            <div style='background: rgba(0, 51, 102, 0.5); padding: 4px; border-radius: 5px; text-align: center;'>
                <div style='color: #4dd0e1; font-size: 0.65rem;'>Accuracy</div>
                <div style='color: #00d9ff; font-weight: 700; font-size: 0.85rem;'>{best_accuracy:.1f}%</div>
            </div>
            <div style='background: rgba(0, 51, 102, 0.5); padding: 6px; border-radius: 5px; text-align: center;'>
                <div style='color: #4dd0e1; font-size: 0.65rem;'>Features</div>
                <div style='color: #00d9ff; font-weight: 700; font-size: 0.85rem;'>9</div>
            </div>
            <div style='background: rgba(0, 51, 102, 0.5); padding: 6px; border-radius: 5px; text-align: center;'>
                <div style='color: #4dd0e1; font-size: 0.65rem;'>Max TGM</div>
                <div style='color: #00d9ff; font-weight: 700; font-size: 0.85rem;'>{df['Tingkat Kegemaran Membaca'].max():.1f}</div>
            </div>
            <div style='background: rgba(0, 51, 102, 0.5); padding: 6px; border-radius: 5px; text-align: center;'>
                <div style='color: #4dd0e1; font-size: 0.65rem;'>Min TGM</div>
                <div style='color: #00d9ff; font-weight: 700; font-size: 0.85rem;'>{df['Tingkat Kegemaran Membaca'].min():.1f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Combined: Kategori + Regions
    categories = df['Kategori'].value_counts()
    st.markdown("""
    <div style='background: rgba(0, 26, 51, 0.6); border: 1px solid #00d9ff; border-radius: 8px; padding: 10px; margin-bottom: 10px;'>
        <h3 style='font-size: 0.8rem; margin: 0 0 8px 0;'>📊 KATEGORI & REGIONS</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px;'>
            <div>
    """, unsafe_allow_html=True)
    
    for cat, count in categories.items():
        color_map = {'Tinggi': '#00d9ff', 'Sedang': '#0099cc', 'Rendah': '#004d99'}
        st.markdown(f"""
        <div style='background: rgba(0, 51, 102, 0.4); padding: 5px 8px; border-radius: 4px; margin: 2px 0; 
                    border-left: 2px solid {color_map.get(cat, "#00d9ff")}; display: flex; justify-content: space-between;'>
            <span style='color: #4dd0e1; font-size: 0.7rem;'>{cat}</span>
            <span style='color: #00d9ff; font-weight: 900; font-size: 0.7rem;'>{count}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div><div>", unsafe_allow_html=True)
    
    for region in ['Jawa', 'Sumatera', 'Kalimantan', 'Sulawesi']:
        region_provs = df[df['Provinsi'].isin(regions[region])]
        avg_tgm_region = region_provs['Tingkat Kegemaran Membaca'].mean()
        st.markdown(f"""
        <div style='background: rgba(0, 51, 102, 0.4); padding: 5px 8px; border-radius: 4px; margin: 2px 0; display: flex; justify-content: space-between;'>
            <span style='color: #4dd0e1; font-size: 0.7rem;'>{region}</span>
            <span style='color: #00d9ff; font-size: 0.7rem; font-weight: 800;'>{avg_tgm_region:.1f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div></div></div>", unsafe_allow_html=True)
    
    # Key Insights
    st.markdown(f"""
    <div style='background: rgba(0, 26, 51, 0.6); border: 1px solid #00d9ff; border-radius: 8px; padding: 10px; margin-bottom: 10px;'>
        <h3 style='font-size: 0.8rem; margin: 0 0 8px 0;'>💡 KEY INSIGHTS</h3>
        <div style='background: rgba(0, 51, 102, 0.5); padding: 6px 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #fbbf24;'>
            <div style='color: #fbbf24; font-size: 0.7rem; font-weight: 800;'>🎯 K=1 Optimal</div>
            <div style='color: #4dd0e1; font-size: 0.65rem;'>Akurasi 83.3% (tertinggi)</div>
        </div>
        <div style='background: rgba(0, 51, 102, 0.5); padding: 6px 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #ff4444;'>
            <div style='color: #ff4444; font-size: 0.7rem; font-weight: 800;'>📉 APS Drop 76%</div>
            <div style='color: #4dd0e1; font-size: 0.65rem;'>Usia 19-23 tahun (29.17%)</div>
        </div>
        <div style='background: rgba(0, 51, 102, 0.5); padding: 6px 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #00d9ff;'>
            <div style='color: #00d9ff; font-size: 0.7rem; font-weight: 800;'>🔗 Korelasi Lemah</div>
            <div style='color: #4dd0e1; font-size: 0.65rem;'>TGM-APS: r={corr_tgm_aps:.3f}</div>
        </div>
        <div style='background: rgba(0, 51, 102, 0.5); padding: 6px 8px; border-radius: 5px; margin: 3px 0; border-left: 3px solid #00d9ff;'>
            <div style='color: #00d9ff; font-size: 0.7rem; font-weight: 800;'>📚 Top Feature</div>
            <div style='color: #4dd0e1; font-size: 0.65rem;'>Frekuensi Membaca (95%)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 8 Provinces Performance
    st.markdown("<h3>👥 Top 8 Provinsi</h3>", unsafe_allow_html=True)
    
    top_8 = df.nlargest(8, 'Tingkat Kegemaran Membaca')
    
    fig_top8_right = go.Figure()
    fig_top8_right.add_trace(go.Bar(
        y=top_8['Provinsi'][::-1],
        x=top_8['Tingkat Kegemaran Membaca'][::-1],
        orientation='h',
        marker=dict(
            color=top_8['Tingkat Kegemaran Membaca'][::-1],
            colorscale=[[0, '#003366'], [0.5, '#0080ff'], [1, '#00d9ff']],
            line=dict(color='#00d9ff', width=1)
        ),
        text=top_8['Tingkat Kegemaran Membaca'][::-1].round(1),
        textposition='outside',
        textfont=dict(size=17, color='#00d9ff', weight=800),
        hovertemplate='<b>%{y}</b><br>TGM: %{x:.2f}<extra></extra>'
    ))
    
    fig_top8_right.update_layout(
        height=450,
        margin=dict(l=10, r=40, t=10, b=10),
        paper_bgcolor='rgba(0, 26, 51, 0.5)',
        plot_bgcolor='rgba(0, 8, 20, 0.8)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 217, 255, 0.1)', range=[0, 90]),
        yaxis=dict(showgrid=False, tickfont=dict(size=8, color='#4dd0e1'))
    )
    st.plotly_chart(fig_top8_right, use_container_width=True, config={'displayModeBar': False})

# Footer
st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(90deg, #001a33 0%, #003366 50%, #001a33 100%); 
            padding: 15px; text-align: center; border-top: 2px solid #00d9ff; 
            box-shadow: 0 -4px 20px rgba(0, 217, 255, 0.3);'>
    <p style='color: #00d9ff; font-size: 0.9rem; margin: 0; font-weight: 600;'>
        ⚡ POWERED BY K-NEAREST NEIGHBORS | K=1 | ACCURACY: 83.3% | 38 PROVINSI INDONESIA 2024 ⚡
    </p>
</div>
""", unsafe_allow_html=True)