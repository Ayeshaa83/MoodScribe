/* ═══════════════════════════════════════════════════════════════════════════
   MoodScribe — Frontend Logic
   Shared JavaScript for auth, API calls, toasts, and constants
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── Supabase Configuration ─────────────────────────────────────────────────
// ⚠️ REPLACE THESE WITH YOUR SUPABASE PROJECT CREDENTIALS
const SUPABASE_URL = "https://jjebekdjsqzridvishhu.supabase.co";
const SUPABASE_ANON = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpqZWJla2Rqc3F6cmlkdmlzaGh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU2NjY1OTMsImV4cCI6MjA5MTI0MjU5M30.aTFYjICL6Npsz6FKF0l0pD_qKPOI1h59GZYTG7YfZEc";

const supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON);


// ─── Emotion Constants ──────────────────────────────────────────────────────
const EMOTION_EMOJIS = {
    joy: "😊",
    sadness: "😢",
    anger: "😠",
    love: "❤️",
    fear: "😨",
    surprise: "😲",
};

const EMOTION_COLORS = {
    joy: "#F59E0B",
    sadness: "#3B82F6",
    anger: "#EF4444",
    love: "#EC4899",
    fear: "#8B5CF6",
    surprise: "#F97316",
};


// ─── Logout ─────────────────────────────────────────────────────────────────
async function logout() {
    await supabaseClient.auth.signOut();
    window.location.href = "index.html";
}


// ─── Toast Notifications ────────────────────────────────────────────────────
function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("toast-exit");
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
