import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import streamlit as st
import matplotlib.pyplot as plt

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
    st.title("Bakery Data Analysis")

    # Load the data
    bakeryDF = load_data()

    st.write("Database dimension :", bakeryDF.shape)
    st.write("Database size      :", bakeryDF.size)
    st.write(bakeryDF.info())

    itemFrequency = bakeryDF['Items'].value_counts().sort_values(ascending=False)
    plot_most_frequent_items(itemFrequency)

    peakHours = bakeryDF.groupby('Daypart')['Items'].count().sort_values(ascending=False)
    plot_peak_selling_hours(peakHours)

    mpd = bakeryDF.groupby('Day')['Items'].count().sort_values(ascending=False)
    plot_productive_day(mpd)

    mpm = bakeryDF.groupby('Month')['Items'].count().sort_values(ascending=False)
    plot_productive_month(mpm)

    transactions = []
    for item in bakeryDF['TransactionNo'].unique():
        lst = list(set(bakeryDF[bakeryDF['TransactionNo'] == item]['Items']))
        transactions.append(lst)

    te = TransactionEncoder()
    encodedData = te.fit(transactions).transform(transactions)
    data = pd.DataFrame(encodedData, columns=te.columns_)

    frequentItems = apriori(data, use_colnames=True, min_support=0.02)
    rules = association_rules(frequentItems, metric="lift", min_threshold=1)
    rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))
    rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))

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

    index_names = rules[rules['consequents'] == 'Coffee'].index
    refinedRules = rules.drop(index_names).sort_values('lift', ascending=False)
    refinedRules.drop(['leverage', 'conviction'], axis=1, inplace=True)
    refinedRules = refinedRules.reset_index()
    st.write(refinedRules)

if __name__ == "__main__":
    main()