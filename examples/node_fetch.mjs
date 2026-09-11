// G69: node client (node >= 18, global fetch).
const BASE = process.env.BASE || "http://127.0.0.1:8080";

async function post(path, payload) {
  const r = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("text/event-stream")) {
    const text = await r.text();
    for (const line of text.split("\n")) {
      if (line.startsWith("data:") && !line.includes("[DONE]")) {
        process.stdout.write(JSON.parse(line.slice(5)).choices[0].text);
      }
    }
    process.stdout.write("\n");
    return;
  }
  return r.json();
}

const health = await fetch(BASE + "/healthz").then((r) => r.json());
console.log("health:", health);
const c = await post("/v1/completions", { prompt: "Hello", max_tokens: 20, temperature: 0 });
console.log("completion:", JSON.stringify(c.choices[0].text.slice(0, 80)));
const ch = await post("/v1/chat/completions", {
  messages: [{ role: "user", content: "Hello" }],
  max_tokens: 20,
});
console.log("chat:", JSON.stringify(ch.choices[0].message.content.slice(0, 80)));
await post("/v1/completions", { prompt: "Hello", max_tokens: 10, stream: true });
