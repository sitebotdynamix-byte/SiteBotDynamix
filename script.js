import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';
import * as webLLM from 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.46/+esm';

// Prevent local model fallback fetching
env.allowLocalModels = false;

/* ==========================================================================
   PART 1: INTERACTIVE UI CONTROLLERS (Hardware Diagnostic & Cost Calculator)
   ========================================================================== */

function initPageInteractions() {
  // --- 1. WebGPU Hardware Diagnostic ---
  const checkGpuBtn = document.getElementById('check-gpu-btn');
  const gpuIcon = document.getElementById('gpu-status-icon');
  const gpuTitle = document.getElementById('gpu-status-title');
  const gpuDesc = document.getElementById('gpu-status-desc');

  async function testWebGPU() {
    if (!navigator.gpu) {
      if (gpuIcon) gpuIcon.innerText = '⚠️';
      if (gpuTitle) gpuTitle.innerText = 'WebGPU Not Supported';
      if (gpuDesc) gpuDesc.innerText = 'Your browser does not currently support WebGPU. The bot will automatically fallback to WebAssembly CPU acceleration.';
      return;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        if (gpuIcon) gpuIcon.innerText = '⚡';
        if (gpuTitle) gpuTitle.innerText = 'WebGPU Acceleration Ready!';
        if (gpuDesc) gpuDesc.innerText = 'Hardware acceleration is active. Your browser is fully capable of running local neural network models at peak performance.';
      } else {
        if (gpuIcon) gpuIcon.innerText = '⚙️';
        if (gpuTitle) gpuTitle.innerText = 'WebGPU Adapter Unavailable';
        if (gpuDesc) gpuDesc.innerText = 'WebGPU API is present, but no compatible graphics adapter was detected.';
      }
    } catch (e) {
      if (gpuIcon) gpuIcon.innerText = '❌';
      if (gpuTitle) gpuTitle.innerText = 'Diagnostic Error';
      if (gpuDesc) gpuDesc.innerText = 'An error occurred while inspecting GPU adapters.';
    }
  }

  if (checkGpuBtn) {
    checkGpuBtn.addEventListener('click', testWebGPU);
  }

  // Auto-run light check on load
  testWebGPU();

  // --- 2. Interactive ROI / Cloud Savings Calculator ---
  const chatsInput = document.getElementById('monthly-chats');
  const msgsInput = document.getElementById('avg-messages');
  const chatsVal = document.getElementById('chats-value');
  const msgsVal = document.getElementById('msg-value');
  const costVal = document.getElementById('projected-cost');

  function calculateSavings() {
    if (!chatsInput || !msgsInput) return;

    const chats = parseInt(chatsInput.value, 10);
    const msgs = parseInt(msgsInput.value, 10);

    if (chatsVal) chatsVal.innerText = chats.toLocaleString();
    if (msgsVal) msgsVal.innerText = msgs;

    // Average cloud API cost calculation (~$0.003 per conversation message turn)
    const totalMessages = chats * msgs;
    const estimatedCost = totalMessages * 0.003;

    if (costVal) {
      costVal.innerText = `$${estimatedCost.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })} / mo`;
    }
  }

  if (chatsInput && msgsInput) {
    chatsInput.addEventListener('input', calculateSavings);
    msgsInput.addEventListener('input', calculateSavings);
    calculateSavings();
  }
}

/* ==========================================================================
   PART 2: CLIENT-SIDE AI ENGINE (RAG + WebGPU Chat Assistant Widget)
   ========================================================================== */

class ClientSideSiteBot {
  constructor() {
    this.extractedDocs = [];
    this.docEmbeddings = [];
    this.embedder = null;
    this.engine = null;
    this.isReady = false;

    // High-efficiency instruction model optimized for WebGPU
    this.selectedModel = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';

    this.initUI();
    this.initAI();
  }

  initUI() {
    const chatContainer = document.createElement('div');
    chatContainer.id = 'sitebot-widget';
    chatContainer.innerHTML = `
      <style>
        #sitebot-toggle {
          position: fixed; bottom: 24px; right: 24px; z-index: 9999;
          background: var(--primary, #2563eb); color: #ffffff; border: none; border-radius: 50px;
          padding: 14px 22px; font-weight: 600; cursor: pointer;
          box-shadow: 0 4px 20px rgba(37,99,235,0.35); font-family: 'Inter', sans-serif;
          transition: transform 0.2s ease, background 0.2s ease;
        }
        #sitebot-toggle:hover { transform: translateY(-2px); background: var(--primary-hover, #1d4ed8); }
        #sitebot-window {
          position: fixed; bottom: 85px; right: 24px; width: 370px; height: 520px;
          background: #ffffff; border: 1px solid var(--border, #e2e8f0); border-radius: var(--radius, 12px);
          box-shadow: 0 12px 30px rgba(0,0,0,0.12); display: none; flex-direction: column;
          z-index: 9999; overflow: hidden; font-family: 'Inter', sans-serif;
        }
        #sitebot-header {
          background: var(--bg-dark, #0f172a); color: #fff; padding: 14px 16px; font-weight: 700;
          font-size: 14px; display: flex; justify-content: space-between; align-items: center;
        }
        #sitebot-messages {
          flex: 1; padding: 14px; overflow-y: auto; font-size: 13px; line-height: 1.5; background: var(--bg-light, #f8fafc);
        }
        .sb-msg { margin-bottom: 12px; max-width: 85%; padding: 10px 14px; border-radius: 10px; word-wrap: break-word; }
        .sb-msg.user { background: var(--primary, #2563eb); color: white; margin-left: auto; border-bottom-right-radius: 2px; }
        .sb-msg.bot { background: #e2e8f0; color: #0f172a; margin-right: auto; border-bottom-left-radius: 2px; }
        #sitebot-input-area { display: flex; border-top: 1px solid var(--border, #e2e8f0); background: #fff; }
        #sitebot-input { flex: 1; border: none; padding: 14px; font-size: 13px; outline: none; }
        #sitebot-send { background: var(--primary, #2563eb); color: white; border: none; padding: 0 18px; font-weight: 600; cursor: pointer; }
        #sitebot-send:disabled { background: #94a3b8; cursor: not-allowed; }
        #sitebot-status { font-size: 11px; color: #94a3b8; padding: 6px 12px; background: var(--bg-dark, #0f172a); text-align: center; border-top: 1px solid #1e293b; }
      </style>
      <button id="sitebot-toggle">💬 Chat AI</button>
      <div id="sitebot-window">
        <div id="sitebot-header">
          <span>SiteBot AI Assistant</span>
          <span id="sitebot-close" style="cursor:pointer; padding: 0 4px;">✕</span>
        </div>
        <div id="sitebot-messages"></div>
        <div id="sitebot-status">Initializing WebGPU Engine...</div>
        <form id="sitebot-input-area">
          <input type="text" id="sitebot-input" placeholder="Ask about this website..." disabled />
          <button type="submit" id="sitebot-send" disabled>Send</button>
        </form>
      </div>
    `;
    document.body.appendChild(chatContainer);

    this.toggleBtn = document.getElementById('sitebot-toggle');
    this.windowEl = document.getElementById('sitebot-window');
    this.closeBtn = document.getElementById('sitebot-close');
    this.messagesEl = document.getElementById('sitebot-messages');
    this.inputEl = document.getElementById('sitebot-input');
    this.sendBtn = document.getElementById('sitebot-send');
    this.statusEl = document.getElementById('sitebot-status');
    this.formEl = document.getElementById('sitebot-input-area');

    this.toggleBtn.onclick = () => this.windowEl.style.display = this.windowEl.style.display === 'flex' ? 'none' : 'flex';
    this.closeBtn.onclick = () => this.windowEl.style.display = 'none';

    this.formEl.onsubmit = (e) => {
      e.preventDefault();
      this.handleUserSubmit();
    };
  }

  extractPageContent() {
    const textNodes = [];
    const elements = document.querySelectorAll('h1, h2, h3, p, li, .card, .benefit-item');

    elements.forEach(el => {
      if (el.closest('#sitebot-widget')) return;
      const text = el.innerText.replace(/\s+/g, ' ').trim();
      if (text.length > 25) {
        textNodes.push(text);
      }
    });

    this.extractedDocs = Array.from(new Set(textNodes));
  }

  async initAI() {
    try {
      this.extractPageContent();
      this.statusEl.innerText = 'Loading vector embedding pipeline...';

      // Load Transformers.js feature extractor (~22MB)
      this.embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

      this.statusEl.innerText = 'Indexing page content...';
      for (const doc of this.extractedDocs) {
        const output = await this.embedder(doc, { pooling: 'mean', normalize: true });
        this.docEmbeddings.push({ text: doc, vector: Array.from(output.data) });
      }

      this.statusEl.innerText = 'Initializing WebGPU LLM...';

      // Initialize WebLLM engine
      this.engine = await webLLM.CreateMLCEngine(this.selectedModel, {
        initProgressCallback: (report) => {
          this.statusEl.innerText = `Loading Model: ${Math.round(report.progress * 100)}%`;
        }
      });

      this.statusEl.innerText = 'Ready (100% Client-Side)';
      this.inputEl.disabled = false;
      this.sendBtn.disabled = false;
      this.isReady = true;

      this.appendMessage('bot', 'Hello! I am running 100% inside your browser. Ask me anything about SiteBotDynamix!');
    } catch (err) {
      console.error(err);
      this.statusEl.innerText = 'WebGPU error or unsupported device.';
      this.appendMessage('bot', 'WebGPU hardware acceleration is required for full in-browser execution.');
    }
  }

  cosineSimilarity(vecA, vecB) {
    let dot = 0.0, normA = 0.0, normB = 0.0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  async retrieveContext(query, topK = 3) {
    const queryEmbed = await this.embedder(query, { pooling: 'mean', normalize: true });
    const queryVec = Array.from(queryEmbed.data);

    const scored = this.docEmbeddings.map(item => ({
      text: item.text,
      score: this.cosineSimilarity(queryVec, item.vector)
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(item => item.text).join('\n---\n');
  }

  async handleUserSubmit() {
    const query = this.inputEl.value.trim();
    if (!query || !this.isReady) return;

    this.inputEl.value = '';
    this.appendMessage('user', query);

    const context = await this.retrieveContext(query);
    const botMsgDiv = this.appendMessage('bot', '...');

    const systemPrompt = `You are an AI support assistant for SiteBotDynamix.
Answer the user question concisely using ONLY the following context from the website:

CONTEXT:
${context}

If the information is not present in the context, politely state that you do not have that specific details.`;

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: query }
    ];

    try {
      const chunks = await this.engine.chat.completions.create({
        messages,
        stream: true
      });

      let responseText = '';
      for await (const chunk of chunks) {
        const delta = chunk.choices[0]?.delta?.content || '';
        responseText += delta;
        botMsgDiv.innerText = responseText;
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
      }
    } catch (e) {
      console.error(e);
      botMsgDiv.innerText = 'Error processing local LLM generation.';
    }
  }

  appendMessage(role, text) {
    const msg = document.createElement('div');
    msg.className = `sb-msg ${role}`;
    msg.innerText = text;
    this.messagesEl.appendChild(msg);
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    return msg;
  }
}

/* ==========================================================================
   INITIALIZATION
   ========================================================================== */

window.addEventListener('DOMContentLoaded', () => {
  initPageInteractions();
  new ClientSideSiteBot();
});
