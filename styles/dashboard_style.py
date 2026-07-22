import streamlit as st


def load_css():

    st.markdown(
        """

        <style>


        /* =========================
           GLOBAL PAGE STYLE
        ========================== */


        .main {

            background-color: #f7f9fc;

        }



        /* =========================
           TITLE STYLE
        ========================== */


        h1 {

            color: #0b3d91;

            font-weight: 700;

        }



        h2, h3 {

            color: #16324f;

        }



        /* =========================
           KPI CARDS
        ========================== */


        div[data-testid="metric-container"] {


            background-color: white;

            border-radius: 12px;

            padding: 15px;

            box-shadow:
            0px 3px 8px rgba(0,0,0,0.08);


        }



        div[data-testid="metric-container"]
        label {


            color: #5b6770;

            font-size: 14px;

        }



        div[data-testid="metric-container"]
        div {


            color: #0b3d91;

            font-size: 28px;

            font-weight: 700;


        }




        /* =========================
           SIDEBAR
        ========================== */


        section[data-testid="stSidebar"] {


            background-color: #0b3d91;


        }



        section[data-testid="stSidebar"]
        * {


            color: white;


        }




        /* =========================
           DATA TABLE
        ========================== */


        .stDataFrame {


            border-radius: 10px;


        }



        /* =========================
           BUTTON STYLE
        ========================== */


        button {


            border-radius: 8px !important;


        }



        </style>


        """,

        unsafe_allow_html=True

    )



def page_header(title, subtitle=None):


    st.title(title)


    if subtitle:


        st.markdown(

            f"""
            <p style="
            color:#5b6770;
            font-size:17px;">
            {subtitle}
            </p>
            """,

            unsafe_allow_html=True

        )