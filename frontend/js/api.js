async function handle(r) {
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg =
        typeof j.detail === "string"
          ? j.detail
          : Array.isArray(j.detail)
            ? j.detail
                .map((e) => `${e.loc?.slice(1).join(" ")}: ${e.msg}`)
                .join("; ")
            : JSON.stringify(j.detail);
    } catch {}
    throw new Error(`${r.status}: ${msg}`);
  }
  return r.json();
}

export const get = (path) => fetch("/api" + path).then(handle);
export const post = (path, body) =>
  fetch("/api" + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  }).then(handle);
