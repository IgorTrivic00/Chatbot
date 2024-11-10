from rag_functionality import rag_func
import streamlit as st

# Prikaži poruku za izbor poglavlja
if "selected_section" not in st.session_state:
    st.write("Izaberi poglavlje:")
    section_titles = [
        "Uvod",
        "Prva lekcija - Bogati ne rade za platu",
        "Druga lekcija - Zašto je važna finansijska pismenost?",
        "Treća lekcija - Gledajte svoja posla",
        "Četvrta lekcija - Istorija poreza i moći korporacija",
        "Peta lekcija - Bogati pronalaze novac",
        "Šesta lekcija - Radite za znanje a ne za novac"
    ]

    for i, title in enumerate(section_titles):
        if st.button(title):
            st.session_state.selected_section = title
            st.session_state.messages = [  # Resetujemo poruke
                {"role": "assistant", "content": f"Dobrodošli u {title}."}
            ]
            break

# Proveri da li je izabrano poglavlje
if "selected_section" in st.session_state:
    selected_section = st.session_state.selected_section
    
    # Prikaži samo poruku dobrodošlice nakon odabira poglavlja
    if len(st.session_state.messages) == 1:
        with st.chat_message("assistant"):
            st.write(f"Dobrodošli u {selected_section}. Kako mogu da vam pomognem?")

    # get user input
    user_prompt = st.chat_input("Postavite pitanje:")

    if user_prompt is not None:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Loading..."):
                    ai_response = rag_func(user_prompt, selected_section)
                    st.write(ai_response)
                    
            new_ai_message = {"role": "assistant", "content": ai_response}
            st.session_state.messages.append(new_ai_message)