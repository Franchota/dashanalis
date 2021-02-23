import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output

df_2016 = pd.read_csv('http://cdn.buenosaires.gob.ar/datosabiertos/datasets/mapa-del-delito/delitos_2016.csv',parse_dates=['fecha'])
df_2017 = pd.read_csv('http://cdn.buenosaires.gob.ar/datosabiertos/datasets/mapa-del-delito/delitos_2017.csv',parse_dates=['fecha'])
df_2018 = pd.read_csv('http://cdn.buenosaires.gob.ar/datosabiertos/datasets/mapa-del-delito/delitos_2018.csv',parse_dates=['fecha'])
df_2019 = pd.read_csv('http://cdn.buenosaires.gob.ar/datosabiertos/datasets/mapa-del-delito/delitos_2019.csv',parse_dates=['fecha'])
print(df_2016.shape)
print(df_2017.shape)
print(df_2018.shape)
print(df_2019.shape)
df = pd.concat([df_2016,df_2017,df_2018,df_2019],axis=0)
print(df.shape)
df = df[df.barrio.isna()== False].copy()
df.isna().sum()
df =df.drop('subtipo_delito',axis = 1)
df['dia_semana_num'] = df.fecha.dt.dayofweek
df['mes_num'] = df.fecha.dt.strftime('%m')
df['ano'] = df.fecha.dt.strftime('%Y')
df['semana'] = df.fecha.dt.week.astype('int')
df['ano_mes'] = df.fecha.dt.strftime('%Y-%m')
df['dia_ano_num'] =df.fecha.dt.dayofyear

dfdummies = pd.get_dummies(df['tipo_delito'])
df = pd.concat([df, dfdummies], axis=1)
df["Hurto"] = df.filter(like="Hurto")
df["Robo"] = df.filter(like="Robo")

m = {
    "1": 'enero',
    "2": 'febrero',
    "3": 'marzo',
    "4": 'abril',
    "5": 'mayo',
    "6": 'junio',
    "7": 'julio',
    "8": 'agosto',
    "9": 'septiembre',
    "10": 'octubre',
    "11": 'noviembre',
    "12": 'diciembre'
}

USERNAME_PASSWORD_PAIRS = [
    ["franbrom", "Abcdefghi9"], ["username", "password"]
]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

fig_ev = df.groupby(['ano_mes', 'ano', 'mes_num'], as_index=False).count().iloc[:,
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]

tabla_aux1 = df.groupby(['ano_mes', 'tipo_delito', ], as_index=False).count().iloc[:, [0, 1, 2, 3, 4]]
tabla_aux3 = df.groupby(['ano', 'mes_num'], as_index=False).count().iloc[:, [0, 1, 2]]
tabla_aux3.columns = ['Año', 'Mes', 'cantidad']
tabla_aux2 = tabla_aux3.pivot_table(index='Mes', columns='Año', values='cantidad', aggfunc='sum')

figura1 = px.line(fig_ev, x='mes_num', y='fecha', color='ano', hover_name='ano_mes',
                  title="Evolución de la cantidad de hechos",
                  labels=dict(mes_num="Mes del año", fecha="Cantidad de casos (miles)", ano="Año"))
figura1.update_layout(title_x=0.5, xaxis_tickmode="linear")

figura2 = px.line(tabla_aux1, x='ano_mes', y='id', color='tipo_delito',
                  title="Evolución de la cantidad de hechos por tipo de delito",
                  labels=dict(ano_mes="Meses por año", id="Cantidad de casos (miles)"))
figura2.update_layout(title_x=0.5)

figura1.update_layout(paper_bgcolor="#eaf2fc")

app = dash.Dash(__name__)
# auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)
server = app.server

anos = df.ano.unique()
mes_num = df.mes_num.unique()

app.layout = html.Div([
    html.H1("Análisis del crimen. CABA",
            style={"color": "#696d72", 'paddingTop': 25, "fontSize": 70, 'textAlign': 'center'}),

    dcc.Tabs(id="tabs", value="tab0", children=[
        dcc.Tab(label="Inicio", id="Inicio", value="tab0", children=[
            html.Div([
                html.Div(dcc.Graph(id="evolucion", figure=figura1)),
                html.Div([html.Div(dcc.Graph(id="heatmap", figure=figura2),
                                   style={'width': '50%', 'vertical-align': 'left', 'display': 'inline-block'}),
                          html.Div(dcc.Graph(id="tipo_delito", figure={"data": [go.Heatmap(
                              z=tabla_aux3["cantidad"],
                              x=tabla_aux3["Año"], name="año",
                              y=tabla_aux3['Mes'],
                              hoverongaps=False,
                              colorscale='Blues'
                          )], "layout": go.Layout(title="Cantidad de delitos por mes y año", title_x=0.5,
                                                  xaxis_tickmode="linear", xaxis_title_text="Año",
                                                  yaxis_title_text="Mes",
                                                  yaxis_tickmode="linear")}),
                                   style={'width': '50%', 'vertical-align': 'right', 'display': 'inline-block'})], )])
        ]),

        dcc.Tab(label='Barrio y delitos', id="tab1", value="tab1", children=[
            html.H5("Año"),
            dcc.Dropdown(id="dropdown", options=[{"label": x, "value": x} for x in anos],
                         value=anos[0], clearable=False),
            dcc.Graph(id="bar-chart")
        ]),

        dcc.Tab(label='Mapa del delito', children=[
            html.H5("Año"),
            dcc.Dropdown(id="dropdown2", options=[{"label": x, "value": x} for x in anos],
                         value=anos[0], clearable=False),
            html.H5("Mes"),
            dcc.Dropdown(id="dropdown3", options=[{"label": m[str(x)].upper(), "value": x} for x in mes_num],
                         value=mes_num[0], clearable=False),
            html.Div(),
            html.H5(""),

            html.Div([html.Div(dcc.Loading(dcc.Graph(id="maps")),
                               style={'width': '80%', 'vertical-align': 'right', 'display': 'inline-block'}),
                      html.Div(dcc.Graph(id="velocimetro"),
                               style={'width': '20%', 'vertical-align': 'left', 'display': 'inline-block'}),
                      ],
                     )

            # dcc.Loading(dcc.Graph(id="maps")))])

        ]),

        dcc.Tab(label='Información', children=[
            dcc.Graph(
                figure={

                }
            )
        ]),

    ])

])


# def evolucion(ano, ano_mes, mes_num):

#           return fig_eva


@app.callback(
    Output("bar-chart", "figure"),
    [Input("dropdown", "value")])
def update_figure(ano):
    filtered_df = df[df["ano"] == ano]
    fig = go.Histogram(x=filtered_df[filtered_df["tipo_delito"] == "Homicidio"]["comuna"], name="Homicidio")
    fig2 = go.Histogram(x=filtered_df[filtered_df["tipo_delito"] == "Lesiones"]["comuna"], name="Lesiones")
    fig3 = go.Histogram(x=filtered_df[filtered_df["tipo_delito"].str.contains("Hurto")]["comuna"], name="Hurto")
    fig4 = go.Histogram(x=filtered_df[filtered_df["tipo_delito"].str.contains("Robo")]["comuna"], name="Robo")
    return {
        'data': [fig, fig2, fig3, fig4],
        'layout': go.Layout(
            xaxis={"type": "linear", 'title': 'Comuna'},
            yaxis={'title': 'Delitos'},
            barmode="group",
            title="Cantidad de delitos por comuna durante el año " + str(ano),
            title_x=0.5, xaxis_tickmode="linear", paper_bgcolor="#eaf2fc")

    }


@app.callback(
    Output("maps", "figure"),
    [Input("dropdown2", "value"),
     Input("dropdown3", "value")])
def update_map(ano, mes_num):
    filtered_df = df[df["ano"] == ano]
    filtered_df = filtered_df[filtered_df["mes_num"] == mes_num]

    fig_map = px.scatter_mapbox(filtered_df, lat="lat", lon="long", center=dict(lat=-34.599722, lon=-58.381944),
                                hover_name="comuna", color="tipo_delito", zoom=11, height=500)

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r": 0, "t": 25, "l": 0, "b": 0})
    fig_map.update_layout(
        title="Ubicación geográfica de los delitos cometidos el mes " + m[str(mes_num)] + " de " + str(ano))
    fig_map.update_layout(paper_bgcolor="#eaf2fc")
    return fig_map
    # fig_map.update_layout(
    # autosize=True,
    # hovermode='closest',

    # center=dict(
    # lat=38.92,
    # lon=-77.07
    # ))


@app.callback(
    Output("velocimetro", "figure"),
    [Input("dropdown2", "value"),
     Input("dropdown3", "value")])
def update_velocimetro(ano, mes_num):
    filtered_df = df[df["ano"] == ano]
    filtered_df = filtered_df[filtered_df["mes_num"] == mes_num]
    fig_vel = go.Figure(go.Indicator(
        mode="gauge+number",

        value=filtered_df.tipo_delito.count(),
        gauge={'axis': {'range': [None, 20000]},
               'steps': [
                   {'range': [0, 10000], 'color': "lightgray"},
                   {'range': [10000, 17000], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 19900}}, )
    )
    fig_vel.update_layout(title="Cantidad total de delitos")
    fig_vel.update_layout(margin={"r": 125, "t": 300})
    fig_vel.update_layout(paper_bgcolor="#eaf2fc")

    # fig_vel.update_traces(align="right")
    # fig_vel.update_traces(delta_position="right")
    # fig_vel.update_traces(gauge_shape="bullet")
    return fig_vel


if __name__ == '__main__':
    app.run_server()
