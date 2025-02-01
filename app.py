import streamlit as st

# Judul halaman
st.title("🎉 Selamat Datang di Proyek Multimodal Batik IR! 🎉")

# Deskripsi proyek
st.write("Halo! Ini adalah proyek yang baru saja dibuat. Yuk, eksplor fitur di sini!")

# Tombol interaktif
if st.button("Klik aku untuk kejutan 🎁"):
    st.balloons()  # Animasi balon
    st.success("Selamat! Kamu berhasil menemukan kejutan pertama! 🎈")
    st.write("Ayo, terus eksplor untuk menemukan hal lain yang seru!")

# Gambar lucu
st.image(
    "https://media.giphy.com/media/l0HlNaQ6gWfllcjDO/giphy.gif",
    caption="Kucing yang super antusias dengan proyek ini! 🐱",
    use_column_width=True,
)

# Teks tambahan
st.write("Semoga harimu menyenangkan! Jangan lupa tersenyum 😄")
