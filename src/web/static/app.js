const $ = (id) => document.getElementById(id);

const titleEl = $("title");
const breadcrumbEl = $("breadcrumb");
const logEl = $("log");

const chipsEl = $("chips");
const btnPredict = $("btnPredict");
const btnConfirm = $("btnConfirm");
const btnClear = $("btnClear");

// 你可以把这里的“推荐关键词”换成你业务常见的
const SUGGEST = [
  "吊带", "短袖", "长袖", "连衣裙", "半身裙", "T恤", "衬衫", "外套",
  "宽松", "修身", "显瘦", "高腰", "百搭", "纯棉", "时尚",
  "粮油速食", "方便面", "零食", "饮料", "牛奶"
];

function renderChips() {
  chipsEl.innerHTML = "";
  SUGGEST.forEach((t) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.type = "button";
    chip.innerText = t;
    chip.onclick = () => {
      const cur = titleEl.value.trim();
      titleEl.value = cur ? (cur + " " + t) : t;
      titleEl.focus();
    };
    chipsEl.appendChild(chip);
  });
}

function setLog(msg, isError=false) {
  logEl.innerText = msg || "";
  logEl.className = isError ? "log err" : "log";
}

// 你现在 /predict 返回的是 label（字符串）
// 如果你想要“面包屑路径”那种展示，可以让后端返回 label_path（数组或字符串），这里也能适配
function formatLabelToBreadcrumb(label) {
  // 暂时：如果 label 本身没有层级，就直接显示
  // 如果你未来返回 "服装鞋包>女装/女士精品>连衣裙" 这种，就会自动变面包屑
  return label || "—";
}

async function predict() {
  const title = titleEl.value.trim();
  if (!title) {
    setLog("请输入商品名称", true);
    return;
  }

  setLog("正在请求 /predict ...");
  breadcrumbEl.innerText = "预测中...";

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ title })
    });

    if (!resp.ok) {
      const txt = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${txt}`);
    }

    const data = await resp.json();
    const label = data.label;

    breadcrumbEl.innerText = formatLabelToBreadcrumb(label);
    setLog("预测完成 ✅");
  } catch (e) {
    breadcrumbEl.innerText = "—";
    setLog("请求失败 ❌ " + (e?.message || e), true);
  }
}

btnPredict.onclick = predict;

btnConfirm.onclick = () => {
  const title = titleEl.value.trim();
  const cat = breadcrumbEl.innerText.trim();
  if (!title) return setLog("请先输入商品名称", true);
  if (!cat || cat === "—" || cat === "预测中...") return setLog("请先点击“推荐类目”得到预测结果", true);

  // 这里你可以改成：把确认结果 POST 到 /confirm 保存到数据库
  navigator.clipboard?.writeText(`${title}\t${cat}`).catch(() => {});
  setLog(`已确认：${cat}（已复制 “标题\\t类目” 到剪贴板）✅`);
};

btnClear.onclick = () => {
  titleEl.value = "";
  breadcrumbEl.innerText = "—";
  setLog("");
};

renderChips();
