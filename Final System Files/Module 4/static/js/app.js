const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");
const statusEl = document.querySelector("#status");
const clearButton = document.querySelector("#clear-button");

let sessionId = localStorage.getItem("mento_session_id") || crypto.randomUUID();
localStorage.setItem("mento_session_id", sessionId);
let lastMentalHealthTopic = localStorage.getItem("mento_last_mental_health_topic") || "";

function addMessage(role, text = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  input.disabled = isBusy;
  statusEl.textContent = isBusy ? "Thinking" : "Ready";
}

function updateMetadata(data) {
  return data;
}

async function parseStream(response, assistantBubble) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const event of events) {
      const line = event.split("\n").find((part) => part.startsWith("data: "));
      if (!line) continue;
      const payload = JSON.parse(line.slice(6));

      if (payload.type === "session") {
        sessionId = payload.session_id;
        localStorage.setItem("mento_session_id", sessionId);
      } else if (payload.type === "metadata") {
        updateMetadata(payload.data);
      } else if (payload.type === "token") {
        assistantBubble.textContent += payload.text;
        messages.scrollTop = messages.scrollHeight;
      } else if (payload.type === "replace") {
        assistantBubble.textContent = payload.text;
      } else if (payload.type === "notice") {
        statusEl.textContent = "Fallback";
      } else if (payload.type === "error") {
        assistantBubble.textContent = payload.message;
      } else if (payload.type === "done") {
        updateMetadata(payload.data);
        if (payload.data && payload.data.mental_health_topic) {
          lastMentalHealthTopic = payload.data.mental_health_topic;
          localStorage.setItem("mento_last_mental_health_topic", lastMentalHealthTopic);
        }
      }
    }
  }
}

function autoresize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

input.addEventListener("input", autoresize);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addMessage("user", message);
  input.value = "";
  autoresize();
  const assistantBubble = addMessage("assistant", "");
  setBusy(true);

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        last_mental_health_topic: lastMentalHealthTopic,
      }),
    });
    await parseStream(response, assistantBubble);
  } catch (error) {
    const message = String(error.message || "").toLowerCase();
    assistantBubble.textContent = message.includes("network") || message.includes("failed to fetch")
      ? "The local Flask connection stopped during this request. Please restart or refresh Mento, then try again."
      : error.message;
  } finally {
    setBusy(false);
    input.focus();
  }
});

clearButton.addEventListener("click", async () => {
  await fetch("/api/chat/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  sessionId = crypto.randomUUID();
  lastMentalHealthTopic = "";
  localStorage.setItem("mento_session_id", sessionId);
  localStorage.removeItem("mento_last_mental_health_topic");
  messages.innerHTML = "";
  addMessage("assistant", "Hello, I'm Mento. What's on your mind today?");
});
