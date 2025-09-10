import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
import librosa
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import smtplib
import ssl
from email.message import EmailMessage
import email.utils
import random

# ---------------- Setup ----------------
st.set_page_config(page_title="MindMate", layout="centered")
analyzer = SentimentIntensityAnalyzer()

# ---------------- Data file setup ----------------
data_path = "data/journal_entries.csv"
os.makedirs("data", exist_ok=True)
if not os.path.exists(data_path):
    df = pd.DataFrame(columns=["Date", "Entry", "Mood", "Score"])
    df.to_csv(data_path, index=False)

# ---------------- Mood Analysis Function ----------------
def analyze_mood(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.5:
        mood = "Happy 😊"
    elif score >= 0.1:
        mood = "Okay 🙂"
    elif score >= -0.1:
        mood = "Neutral 😐"
    elif score >= -0.5:
        mood = "Sad 😔"
    else:
        mood = "Very Low 😞"
    return mood, score

# ---------------- Page Title ----------------
st.title("🧠 MindMate")
st.markdown("Your personal mental health companion 💙")

# ---------------- Voice Mood Analyzer ----------------
class VoiceMoodAnalyzer(AudioProcessorBase):
    def __init__(self) -> None:
        self.volume = 0.0
        self.pitch = 0.0

    def recv(self, frame):
        audio = frame.to_ndarray()
        self.volume = np.linalg.norm(audio) / len(audio) if len(audio) > 0 else 0.0
        try:
            y = audio.astype(np.float32)
            sr = 44100  # assume 44.1kHz sample rate
            pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
            self.pitch = np.max(pitches) if pitches.size > 0 else 0.0
        except Exception as e:
            print(f"Error in pitch analysis: {e}")
            self.pitch = 0.0
        return frame

def analyze_mood_based_on_audio(volume, pitch):
    if volume > 0.1 and pitch > 150:
        return "Energetic 😊"
    elif volume > 0.05 and pitch > 100:
        return "Excited 🙂"
    elif volume > 0.05:
        return "Calm 🙂"
    elif volume <= 0.01 and pitch <= 50:
        return "Sad 😔"
    else:
        return "Neutral 😐"

# ---------------- Voice Mood UI ----------------
st.header("🎙 Voice Mood Check")
ctx = webrtc_streamer(
    key="voice-mood-analyzer",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=VoiceMoodAnalyzer,
    media_stream_constraints={"audio": True, "video": False},
)

if ctx.audio_processor:
    volume = ctx.audio_processor.volume
    pitch = ctx.audio_processor.pitch
    mood = analyze_mood_based_on_audio(volume, pitch)
    st.write(f"🎧 Detected Mood: {mood}")

# ---------------- Text Mood Check ----------------
st.header("🌤 How are you feeling today?")
mood_input = st.text_area("Describe your thoughts or mood:", height=100)
if st.button("📝 Submit Entry"):
    if mood_input.strip():
        mood, score = analyze_mood(mood_input)
        st.success(f"Detected Mood: {mood} (Score: {round(score, 2)})")

        new_entry = pd.DataFrame([[datetime.now(), mood_input, mood, score]],
                                 columns=["Date", "Entry", "Mood", "Score"])
        new_entry.to_csv(data_path, mode='a', header=False, index=False)
        df = pd.read_csv(data_path)
    else:
        st.warning("Please write something before submitting.")

# ---------------- Load data ----------------
df = pd.read_csv(data_path)
if "Score" not in df.columns:
    df["Score"] = None

# ---------------- Mood History ----------------
st.header("📖 Your Journal Mood History")
if not df.empty:
    df["Date"] = pd.to_datetime(df["Date"])
    st.dataframe(df.tail(5)[["Date", "Mood", "Entry"]])

    df["DateTime"] = pd.to_datetime(df["Date"])
    df = df.sort_values("DateTime")

    def get_emoji(score):
        if score >= 0.5:
            return "😄"
        elif score >= 0.1:
            return "😊"
        elif score >= -0.1:
            return "😐"
        elif score >= -0.5:
            return "😟"
        else:
            return "😢"

    def get_color(score):
        if score >= 0.5:
            return 'darkgreen'
        elif score >= 0.1:
            return 'green'
        elif score >= -0.1:
            return 'yellow'
        elif score >= -0.5:
            return 'orange'
        else:
            return 'red'

    df["Emoji"] = df["Score"].apply(lambda x: get_emoji(x) if pd.notnull(x) else "")
    df["Color"] = df["Score"].apply(lambda x: get_color(x) if pd.notnull(x) else "black")

    fig, ax = plt.subplots(figsize=(10, 4))
    valid_scores = df[df["Score"].notnull()]
    ax.plot(valid_scores["DateTime"], valid_scores["Score"], color="lightgray", linestyle='--', linewidth=1)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    for i in range(len(df)):
        if pd.notnull(df["Score"].iloc[i]):
            ax.text(df["DateTime"].iloc[i], df["Score"].iloc[i], df["Emoji"].iloc[i],
                    fontsize=14, ha='center', va='center', color=df["Color"].iloc[i])

    ax.set_title("Mood Sentiment Over Time")
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Timestamp")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

    # ---------------- Mood Streak ----------------
    st.subheader("🌟 Mood Streak")
    streak = 0
    last_mood = None
    for mood in df["Mood"][::-1]:
        if last_mood is None or mood == last_mood:
            streak += 1
            last_mood = mood
        else:
            break
    st.info(f"You have been in the mood: {last_mood} for {streak} consecutive entries!")

    last_entry = df["Date"].max()
    if (datetime.now() - last_entry).days >= 2:
        st.warning("⏰ It's been a while since your last journal. Consider checking in today!")
else:
    st.info("No journal entries yet. Start by writing your thoughts!")

# ---------------- AI Mood Tips ----------------
st.header("🧠 AI Mood Insight")
if not df.empty:
    latest_score = df.iloc[-1]["Score"]
    if pd.notnull(latest_score):
        if latest_score >= 0.5:
            tip = "Keep up the positive energy! Maybe spread the happiness to someone else today."
        elif latest_score >= 0.1:
            tip = "You're doing okay — how about a short walk or break to recharge?"
        elif latest_score >= -0.1:
            tip = "It's a neutral day — perhaps reflect on one thing you're thankful for."
        elif latest_score >= -0.5:
            tip = "Feeling down? Try journaling more or talking to a friend."
        else:
            tip = "It’s okay to have low days. Consider deep breathing, rest, or professional support."
        st.info(f"💡 Tip: {tip}")
    else:
        st.info("Your last entry does not have a mood score. Write a new entry to receive personalized tips!")
else:
    st.info("Write a journal entry to receive personalized tips!")

# ---------------- Emergency Info ----------------
st.markdown("---")
st.caption("⚠ This tool is not a substitute for professional therapy. If you're in crisis, please contact a mental health helpline.")
st.markdown("<h3 style='color:#ff4b4b;'>🧠 Mental Health Support Services</h3>", unsafe_allow_html=True)
st.markdown("""
<style>
.emergency-box { background-color: #2c2c2c; border: 1px solid #444; border-radius: 8px; padding: 12px 20px; margin: 10px 0; transition: 0.3s; }
.emergency-box:hover { background-color: #3a3a3a; }
.emergency-label { color: white; font-weight: 600; font-size: 16px; }
.emergency-number { color: #ff4b4b; font-weight: 600; }
</style>
<p style='color:#ccc;'>If you or someone you know is experiencing a mental health crisis, please reach out to the following helplines:</p>
<div class="emergency-box"><span class="emergency-label">📞 iCall Psychosocial Helpline (TISS): </span><span class="emergency-number">+91 9152987821</span></div>
<div class="emergency-box"><span class="emergency-label">📞 AASRA (24/7 Suicide Prevention): </span><span class="emergency-number">+91 9820466726</span></div>
<div class="emergency-box"><span class="emergency-label">📞 Snehi Mental Health Support: </span><span class="emergency-number">+91 9582208181</span></div>
<div class="emergency-box"><span class="emergency-label">📞 Vandrevala Foundation Helpline: </span><span class="emergency-number">1860 266 2345 / 9999 666 555</span></div>
<div class="emergency-box"><span class="emergency-label">📞 Fortis Stress Helpline: </span><span class="emergency-number">+91 8376804102</span></div>
""", unsafe_allow_html=True)

# ---------------- Token Generator & Email ----------------
st.title("🎫 Token Generator")
name = st.text_input("Enter your name")
user_email = st.text_input("Enter your email")
age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)

# Load sender credentials from Streamlit secrets
sender_email = st.secrets["email"]["sender"]
sender_password = st.secrets["email"]["password"]  # This must be a Gmail App Password!
doctor_email = "vismayiakula3@gmail.com"

def send_email(receiver_email, subject, body):
    """Send email using Gmail SMTP with App Password."""
    try:
        msg = EmailMessage()
        msg["From"] = email.utils.formataddr(("MindMate Team", sender_email))
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.set_content(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, sender_password)  # App password here
            server.send_message(msg)

        st.success(f"✅ Email sent to {receiver_email}")
    except Exception as e:
        st.error(f"❌ Could not send email to {receiver_email}: {e}")

if st.button("Generate Token"):
    if name and user_email and age:
        token_id = f"TOK{random.randint(1000,9999)}"
        location = random.choice(["Delhi - Centre A", "Mumbai - Clinic B", "Online (Zoom)", "Hyderabad - Facility C"])

        # Email to user
        subject_user = "🧠 Your MindMate Consultation Token"
        body_user = f"""
Hi {name},

Your token has been successfully generated to meet a MindMate Consultant.

🧾 Token ID: {token_id}
📍 Location: {location}
📅 Date: Within the next 48 hours (you will be contacted shortly)

Please stay available on this email for updates.

Warm regards,  
MindMate Team
"""
        # Email to doctor
        subject_doc = "📥 New Patient Token Notification"
        body_doc = f"""
Hello Doctor,

A new token has been generated by a patient.

🧾 Token ID: {token_id}
👤 Name: {name}
🎂 Age: {age}
📧 Email: {user_email}
📍 Location: {location}
🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,  
MindMate System
"""

        # Send emails
        send_email(user_email, subject_user, body_user)
        send_email(doctor_email, subject_doc, body_doc)
    else:
        st.warning("Please fill in all the details.")
