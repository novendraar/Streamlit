import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import matplotlib.pyplot as plt
import io

# Function to display bar plot for most frequent items
def plot_most_frequent_items(itemFrequency):
    fig = px.bar(itemFrequency.head(20), title='20 Most Frequent Items', color=itemFrequency.head(20), color_continuous_scale=px.colors.sequential.Mint)
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20), xaxis_tickangle=-45, plot_bgcolor='white', coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False, title=' ')
    fig.update_xaxes(title=' ')
    fig.update_traces(texttemplate='%{y}', textposition='outside', hovertemplate='<b>%{x}</b><br>No. of Transactions: %{y}')
    st.plotly_chart(fig)

# Function to display pie chart for peak selling hours
def plot_peak_selling_hours(peakHours):
    fig = go.Figure(data=[go.Pie(labels=['Afternoon', 'Morning', 'Evening', 'Night'],
                values=peakHours, title="Peak Selling Hours", titlefont=dict(size=20), textinfo='label+percent', marker=dict(colors=px.colors.sequential.Mint), hole=.5)])
    fig.update_layout(margin=dict(t=40, b=40, l=0, r=0), font=dict(size=13), showlegend=False)
    st.plotly_chart(fig)

# Function to display bar plot for productive day
def plot_productive_day(mpd):
    fig = px.bar(mpd, title='Most Productive Day', color=mpd, color_continuous_scale=px.colors.sequential.Mint)
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20), xaxis_tickangle=0, plot_bgcolor='white', coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False, title=' ')
    fig.update_xaxes(title=' ')
    fig.update_traces(texttemplate='%{y}', textposition='outside', hovertemplate='<b>%{x}</b><br>No. of Transactions: %{y}')
    st.plotly_chart(fig)

# Function to display bar plot for productive month
def plot_productive_month(mpm):
    fig = px.bar(mpm, title='Most Productive Month', color=mpm, color_continuous_scale=px.colors.sequential.Mint)
    fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), titlefont=dict(size=20), xaxis_tickangle=0, plot_bgcolor='white', coloraxis_showscale=False)
    fig.update_yaxes(showticklabels=False, title=' ')
    fig.update_xaxes(title=' ')
    fig.update_traces(texttemplate='%{y}', textposition='outside', hovertemplate='<b>%{x}</b><br>No. of Transactions: %{y}')
    st.plotly_chart(fig)

# Function to display network graph
def plot_network_graph(G):
    edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(x=[], y=[], text=[], mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='Burg', reversescale=True, color=[], size=15,
        colorbar=dict(thickness=10, title='Node Connections', xanchor='left', titleside='right')))

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = str(adjacencies[0]) + '<br>No of Connections: {}'.format(str(len(adjacencies[1])))
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(title='Item Connections Network', titlefont=dict(size=20),
        plot_bgcolor='white', showlegend=False, margin=dict(b=0, l=0, r=0, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    st.plotly_chart(fig)

# Load and process data
def load_data():
    df = pd.read_csv('Bakery.csv')
    dateTime = pd.to_datetime(df['DateTime'])
    df['Day'] = dateTime.dt.day_name()
    df['Month'] = dateTime.dt.month_name()
    df['Year'] = dateTime.dt.year
    return df

def main():
    st.markdown("""
                <style>
                    .bold-text { font-weight: bold; }
                    .custome-header { font-size: 39px; font-family: 'Arial'; color: #4682B4; }
                    .caption-kecil { font-size: 35px; font-family: 'Arial';} 
                    .header-daftar { font-size: 20px; font-family: 'Arial';}
                    .list-daftar {font-size: 17px; font-family: 'Arial'; margin= 0; padding= 0;}
                    .centered { text-align: center; }
                    .header-sub { font-size: 35px; font-family: 'Arial'; color: #cf9823;}
                    .list-daftar {font-size: 17px; font-family: 'Arial'; margin= 0; padding= 0;}
                </style>
                <h1 class="custome-header bold-text no-space">Project Association with Streamlit</h1>
                <h2 class="caption-kecil bold-text no-space">Big Data FGA ITI 2024</h2>
                <hr style="border: 1px solid #000; width: 100%;"/>
                """, unsafe_allow_html=True)
    st.markdown("""
                <p class ="header-daftar bold-text">Daftar Anggota Tim:</p>
                <p class ="list-daftar bold-text">1. Marsheila Prima Andika <br>2. Megawati Aswiya Putri <br>3. Jilaan Hani Safitri <br>4. Novendra Aliffian Ramadhan</p>
    """, unsafe_allow_html=True)
    st.markdown("""
                <div class="centered">
                    <h1>Bakery Data Analysis</h1>
                </div>
    """, unsafe_allow_html=True)
    st.image("gmbr/bread.jpg")
    st.markdown("""
                <h1 class="header-sub">Project Understanding</h1>
                <h2 class="bold-text">1. Business Understanding</h2>
                <p>Analisis Market Basket:<br>
                - Meneliti kumpulan item untuk menemukan hubungan antar item dalam konteks bisnis.<br>
                - Menggunakan teknik Association Rule Mining.<br>
                Algoritma Apriori:<br>
                - Digunakan dalam Association Rule Mining untuk menemukan pola frekuensi itemset yang sering muncul bersama dalam transaksi.<br>
                Model analisis yang digunakan dalam Market Basket Analysis adalah Association Rule Mining dengan algoritma Apriori.<br>
                <h2 class="bold-text">2. Data Understanding</h2>
                <p>Dataset ini milik "The Bread Basket", sebuah toko roti yang berlokasi di Edinburgh. Dataset ini menyediakan detail transaksi pelanggan yang memesan berbagai item dari toko roti ini secara daring selama periode waktu dari tahun 2016 dan 2017 dari toko roti daring. Dataset ini memiliki 20507 entri, lebih dari 9000 transaksi, dan 4 kolom.<br></p>
                <p>Data diambil dari kaggle sebagai berikut : https://www.kaggle.com/code/akashdeepkuila/market-basket-analysis</p>
                <h2 class="bold-text">3. Teori Asosiasi</h2>
                <h4>1. Support</h4>
                <p>Mengukur frekuensi kemunculan item atau itemset. <br>Rumus => Support(A) = (Jumlah transaksi yang mengandung A) / (Total transaksi</p>
                <h4>2. Confidence</h4>
                <p>Mengukur seberapa sering item B muncul dalam transaksi yang mengandung item A. <br>Rumus => Confidence(A→B) = Support(A ∪ B) / Support(A).</p>
                <h4>3. Lift</h4>
                <p>Mengukur kekuatan aturan dengan memperhitungkan frekuensi kemunculan item B. <br>Rumus => Lift(A→B) = Confidence(A→B) / Support(B).</p>
                """, unsafe_allow_html=True)
    # Load the data
    bakeryDF = load_data()
    st.markdown("""
                <h2 class="bold-text">Data Preparation</h2>
                <h4>Tabel Dataframe Bakery</h4>
                """, unsafe_allow_html=True)
    st.dataframe(bakeryDF)
    st.write("Database dimension :", bakeryDF.shape)
    st.write("Database size      :", bakeryDF.size)
    st.write("Data Frame Info:")
    buffer = io.StringIO()
    bakeryDF.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.markdown("""
                <h1 class="header-sub">Data Exploration and Visualization</h1>
                <h2 class="bold-text">Visualization by Items</h2>
                """, unsafe_allow_html=True)
    itemFrequency = bakeryDF['Items'].value_counts().sort_values(ascending=False)
    plot_most_frequent_items(itemFrequency)
    st.markdown("""
                <h2 class="bold-text">Visualization by Hours</h2>
                """, unsafe_allow_html=True)
    peakHours = bakeryDF.groupby('Daypart')['Items'].count().sort_values(ascending=False)
    plot_peak_selling_hours(peakHours)
    st.markdown("""
                <h2 class="bold-text">Visualization by Days</h2>
                """, unsafe_allow_html=True)
    mpd = bakeryDF.groupby('Day')['Items'].count().sort_values(ascending=False)
    plot_productive_day(mpd)
    st.markdown("""
                <h2 class="bold-text">Visualization by Months</h2>
                """, unsafe_allow_html=True)
    mpm = bakeryDF.groupby('Month')['Items'].count().sort_values(ascending=False)
    plot_productive_month(mpm)

    transactions = []
    for item in bakeryDF['TransactionNo'].unique():
        unique_items = set(bakeryDF[bakeryDF['TransactionNo'] == item]['Items'])
        lst = list(unique_items)
        transactions.append(lst)

    te = TransactionEncoder()
    encodedData = te.fit(transactions).transform(transactions)
    data = pd.DataFrame(encodedData, columns=te.columns_)
    st.markdown("""
                <h1 class="header-sub">Association Rules Generation</h1>
                <h2 class="bold-text">Tabel Frekuensi Item</h2>
                """, unsafe_allow_html=True)
    frequentItems = apriori(data, use_colnames=True, min_support=0.02)
    st.dataframe(frequentItems)
    st.markdown("""
                <h2 class="bold-text">Tabel Rules Asosiasi</h2>
                """, unsafe_allow_html=True)   
    rules = association_rules(frequentItems, metric="lift", min_threshold=1)
    rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
    rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))
    st.dataframe(rules)
    st.markdown("""
                <h2 class="bold-text">Rules Visualization</h2>
                """, unsafe_allow_html=True) 
    network_A = list(rules["antecedents"].unique())
    network_B = list(rules["consequents"].unique())
    node_list = list(set(network_A + network_B))
    G = nx.Graph()
    for i in node_list:
        G.add_node(i)
    for i, j in rules.iterrows():
        G.add_edges_from([(j["antecedents"], j["consequents"])])

    pos = nx.spring_layout(G, k=0.5, dim=2, iterations=400)
    for n, p in pos.items():
        G.nodes[n]['pos'] = p

    plot_network_graph(G)
    st.markdown("""
                <h2 class="bold-text">Refining Rules</h2>
                <p>Confidence untuk consequent yang paling sering muncul selalu memiliki nilai yang tinggi, walaupun asosiasinya sangat rendah, sehingga memberikan jawaban yang rancu. <br>Karena kopi merupakan produk best seller dan dapat direkomendasikan bersama produk lainnya, maka kita akan drop rules tersebut untuk menemukan rules lain yang tidak diketahui.</p>
                """, unsafe_allow_html=True)  
    index_names = rules[rules['consequents'] == 'Coffee'].index
    refinedRules = rules.drop(index_names).sort_values('lift', ascending=False)
    refinedRules.drop(['leverage', 'conviction'], axis=1, inplace=True)
    refinedRules = refinedRules.reset_index()
    st.write(refinedRules)

    st.markdown("""
                <h1 class="header-sub">Summary</h1>
                <h2 class="bold-text">Insights</h2>
                <p class="bold-text">- Kopi merupakan produk best seller di toko roti ini dan kopi menunjukkan asosiasi dengan 8 produk lainnya.<br>- Lebih dari 11% pecinta kopi juga membeli kue, sedangkan 10% lainnya membeli pastry.<br>- Lebih dari 16% pecinta teh akan membeli kue, sedangkan lebih dari 20% pecinta kue akan membeli teh.<br>- Lebih dari 33% pecinta pastry akan membeli roti, sedangkan hanya hampir 9% pembeli pastry akan membeli roti</p>
                <h2 class="bold-text">Business Strategy</h2>
                <p>Terdapat beberapa strategi yang dapat dilakukan oleh toko roti untuk meningkatkan penjualan berdasarkan hasil analisis asosiasi yang telah dilakukan antara kopi dan 8 produk lainnya, yaitu:<br></p>
                <p class="bold-text">- Promosi diskon pada produk-produk tersebut dapat menarik pelanggan untuk membeli kopi atau sebaliknya.<br>- Menata produk-produk tersebut di dekat konter pemesanan kopi untuk menggoda pelanggan agar membelinya</p>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()