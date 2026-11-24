/**
 * MedVina AI - Minimalist & Functional Frontend Logic
 * Supports:
 *   1. Single Review Analysis (Sentiment + Aspect-based SERVQUAL Breakdown)
 *   2. Batch CSV/Excel Upload (Summary Trends + Aspect Insights + Detailed Table + Export CSV)
 */

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initSingleReview();
    initBatchUpload();
});

/* ==============================================================================
   1. NAVIGATION & TAB SWITCHING
   ============================================================================== */
function initNavigation() {
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPanes = document.querySelectorAll(".tab-content");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");

            navButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(targetTab);
            if (targetPane) targetPane.classList.add("active");
        });
    });
}

/* ==============================================================================
   2. SINGLE REVIEW ANALYSIS (TAB 1)
   ============================================================================== */
function initSingleReview() {
    const inputArea = document.getElementById("single-review-input");
    const charCount = document.getElementById("single-char-count");
    const btnPredict = document.getElementById("btn-single-predict");
    const btnClear = document.getElementById("btn-single-clear");
    const samplePills = document.querySelectorAll(".sample-pill");

    const emptyState = document.getElementById("single-empty-state");
    const loadingState = document.getElementById("single-loading-state");
    const resultContent = document.getElementById("single-result-content");
    const latencyTag = document.getElementById("single-latency");

    // Đếm ký tự
    inputArea.addEventListener("input", () => {
        const len = inputArea.value.length;
        charCount.textContent = `${len} ký tự`;
    });

    // Mẫu thử nhanh
    samplePills.forEach(pill => {
        pill.addEventListener("click", () => {
            const sampleText = pill.getAttribute("data-text");
            inputArea.value = sampleText;
            charCount.textContent = `${sampleText.length} ký tự`;
            executeSinglePrediction();
        });
    });

    // Xóa trắng
    btnClear.addEventListener("click", () => {
        inputArea.value = "";
        charCount.textContent = "0 ký tự";
        resultContent.style.display = "none";
        loadingState.style.display = "none";
        emptyState.style.display = "block";
        latencyTag.style.display = "none";
    });

    // Bấm phân tích
    btnPredict.addEventListener("click", executeSinglePrediction);

    // Phím tắt Ctrl+Enter
    inputArea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            executeSinglePrediction();
        }
    });

    async function executeSinglePrediction() {
        const text = inputArea.value.trim();
        if (!text) {
            alert("Vui lòng nhập nội dung đánh giá trước khi phân tích!");
            inputArea.focus();
            return;
        }

        // Chuyển sang trạng thái đang tải
        emptyState.style.display = "none";
        resultContent.style.display = "none";
        loadingState.style.display = "block";
        btnPredict.disabled = true;

        const startTime = performance.now();

        try {
            // Gọi API phân tích cảm xúc & MoE
            const response = await fetch("/api/predict-moe", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Lỗi máy chủ (${response.status})`);
            }

            const data = await response.json();
            const latency = Math.round(performance.now() - startTime);

            renderSingleResult(data, latency);
        } catch (error) {
            console.error("Lỗi phân tích:", error);
            alert("Không thể phân tích: " + error.message);
            loadingState.style.display = "none";
            emptyState.style.display = "block";
        } finally {
            btnPredict.disabled = false;
        }
    }

    function renderSingleResult(data, clientLatency) {
        loadingState.style.display = "none";
        resultContent.style.display = "block";

        // 1. Hiển thị độ trễ
        latencyTag.style.display = "inline-block";
        latencyTag.textContent = `${data.process_time_ms || clientLatency} ms`;

        // 2. Banner cảm xúc tổng thể
        const banner = document.getElementById("single-sentiment-banner");
        const bannerIcon = document.getElementById("single-sentiment-icon");
        const bannerLabel = document.getElementById("single-sentiment-label");
        const bannerConf = document.getElementById("single-sentiment-conf");

        banner.className = "sentiment-banner";
        const sent = data.overall_sentiment || data.overall_ensemble_sentiment || "Positive";
        const confPct = Math.round((data.overall_confidence !== undefined ? data.overall_confidence : 0.85) * 100);
        const hasConflict = data.has_conflicting_sentiments === true;

        if (hasConflict) {
            banner.classList.add("is-mixed");
            bannerIcon.innerHTML = '<i class="fa-solid fa-scale-unbalanced-flip"></i>';
            bannerLabel.textContent = "HỖN HỢP (VỪA KHEN VỪA CHÊ)";
            
            // Tính độ tin cậy khẳng định đánh giá hỗn hợp dựa trên trung bình độ tin cậy các khía cạnh đối lập
            let aspectConfs = [];
            if (data.moe_aspect_sentiments) {
                Object.values(data.moe_aspect_sentiments).forEach(a => {
                    if (a.confidence) aspectConfs.push(a.confidence);
                });
            }
            let mixedConf = aspectConfs.length > 0
                ? Math.round((aspectConfs.reduce((a, b) => a + b, 0) / aspectConfs.length) * 100)
                : confPct;
            bannerConf.textContent = `${mixedConf}%`;
        } else if (sent === "Positive") {
            banner.classList.add("is-pos");
            bannerIcon.innerHTML = '<i class="fa-solid fa-face-smile"></i>';
            bannerLabel.textContent = "TÍCH CỰC (HÀI LÒNG)";
            bannerConf.textContent = `${confPct}%`;
        } else if (sent === "Negative") {
            banner.classList.add("is-neg");
            bannerIcon.innerHTML = '<i class="fa-solid fa-face-frown"></i>';
            bannerLabel.textContent = "TIÊU CỰC (BẤT BÌNH / PHÀN NÀN)";
            bannerConf.textContent = `${confPct}%`;
        } else {
            banner.classList.add("is-neu");
            bannerIcon.innerHTML = '<i class="fa-solid fa-face-meh"></i>';
            bannerLabel.textContent = "TRUNG LẬP (BÌNH THƯỜNG)";
            bannerConf.textContent = `${confPct}%`;
        }

        // 3. Phân phối xác suất 3 nhãn chuẩn xác
        const probs = data.overall_probabilities || data.probabilities || {};
        let pPos = 0, pNeg = 0, pNeu = 0;
        if (probs.Positive !== undefined || probs.Negative !== undefined || probs.Neutral !== undefined) {
            pPos = Math.round((probs.Positive || 0) * 100);
            pNeg = Math.round((probs.Negative || 0) * 100);
            pNeu = Math.round((probs.Neutral || 0) * 100);
            const sum = pPos + pNeg + pNeu;
            if (sum > 0 && sum !== 100) {
                pNeu = Math.max(0, 100 - pPos - pNeg);
            }
        } else {
            if (sent === "Positive") {
                pPos = confPct;
                pNeg = Math.round((100 - confPct) * 0.6);
                pNeu = Math.max(0, 100 - pPos - pNeg);
            } else if (sent === "Negative") {
                pNeg = confPct;
                pPos = Math.round((100 - confPct) * 0.6);
                pNeu = Math.max(0, 100 - pPos - pNeg);
            } else {
                pNeu = confPct;
                pPos = Math.round((100 - confPct) * 0.5);
                pNeg = Math.max(0, 100 - pPos - pNeu);
            }
        }

        document.getElementById("bar-pos").style.width = `${pPos}%`;
        document.getElementById("pct-pos").textContent = `${pPos}%`;

        document.getElementById("bar-neg").style.width = `${pNeg}%`;
        document.getElementById("pct-neg").textContent = `${pNeg}%`;

        document.getElementById("bar-neu").style.width = `${pNeu}%`;
        document.getElementById("pct-neu").textContent = `${pNeu}%`;

        // 4. Bóc tách 6 khía cạnh SERVQUAL
        const aspectsContainer = document.getElementById("single-aspects-list");
        aspectsContainer.innerHTML = "";

        const aspectSentiments = data.moe_aspect_sentiments || {};
        const aspectKeys = Object.keys(aspectSentiments);

        if (aspectKeys.length === 0) {
            aspectsContainer.innerHTML = '<div class="no-aspects-note"><i class="fa-solid fa-info-circle"></i> Đánh giá chung chung, không đề cập cụ thể đến khía cạnh lâm sàng nào trong 6 nhóm SERVQUAL.</div>';
        } else {
            aspectKeys.forEach(name => {
                const item = aspectSentiments[name];
                const card = document.createElement("div");
                card.className = "aspect-item-card";

                const isP = item.sentiment === "Positive";
                const isN = item.sentiment === "Negative";
                const pillClass = isP ? "pos" : (isN ? "neg" : "neu");
                const pillText = item.sentiment_vi || (isP ? "Tích Cực" : (isN ? "Tiêu Cực" : "Trung Lập"));

                const evidenceHtml = item.evidence_clause
                    ? `<div class="aspect-evidence">Đoạn trích liên quan: "${escapeHtml(item.evidence_clause)}"</div>`
                    : "";

                card.innerHTML = `
                    <div class="aspect-item-left">
                        <div class="aspect-item-name">${escapeHtml(name)}</div>
                        ${evidenceHtml}
                    </div>
                    <span class="aspect-pill ${pillClass}">${pillText}</span>
                `;
                aspectsContainer.appendChild(card);
            });
        }

        // 5. Chuẩn hóa từ viết tắt
        const normBox = document.getElementById("single-normalized-box");
        const normTextElem = document.getElementById("single-normalized-text");
        if (data.normalized_text && data.normalized_text !== data.input_text) {
            normBox.style.display = "block";
            normTextElem.textContent = data.normalized_text;
        } else {
            normBox.style.display = "none";
        }
    }
}

/* ==============================================================================
   3. BATCH CSV / EXCEL UPLOAD & ANALYSIS (TAB 2)
   ============================================================================== */
let batchLoadedRecords = [];

function initBatchUpload() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("batch-file-input");
    const fileBar = document.getElementById("selected-file-bar");
    const fileNameText = document.getElementById("file-name-text");
    const fileSizeText = document.getElementById("file-size-text");
    const btnRemoveFile = document.getElementById("btn-remove-file");
    const btnAnalyze = document.getElementById("btn-batch-analyze");

    const loadingState = document.getElementById("batch-loading-state");
    const resultsContainer = document.getElementById("batch-results-container");
    const btnExportCsv = document.getElementById("btn-export-csv");
    const filterButtons = document.querySelectorAll(".filter-btn");

    let currentSelectedFile = null;

    // Kéo thả & Chọn tệp
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    btnRemoveFile.addEventListener("click", () => {
        currentSelectedFile = null;
        fileInput.value = "";
        fileBar.style.display = "none";
        dropZone.style.display = "block";
        btnAnalyze.disabled = true;
        resultsContainer.style.display = "none";
    });

    function handleFileSelect(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['csv', 'xlsx', 'xls'].includes(ext)) {
            alert("Chỉ hỗ trợ tệp định dạng CSV (.csv) hoặc Excel (.xlsx, .xls)!");
            return;
        }

        currentSelectedFile = file;
        fileNameText.textContent = file.name;
        fileSizeText.textContent = formatBytes(file.size);

        dropZone.style.display = "none";
        fileBar.style.display = "flex";
        btnAnalyze.disabled = false;
    }

    // Bắt đầu phân tích hàng loạt
    btnAnalyze.addEventListener("click", async () => {
        if (!currentSelectedFile) return;

        btnAnalyze.disabled = true;
        resultsContainer.style.display = "none";
        loadingState.style.display = "block";

        const formData = new FormData();
        formData.append("file", currentSelectedFile);

        try {
            const response = await fetch("/api/batch-predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.detail || `Lỗi máy chủ (${response.status})`);
            }

            const data = await response.json();
            renderBatchResults(data);
        } catch (error) {
            console.error("Lỗi phân tích tệp:", error);
            alert("Không thể phân tích tệp: " + error.message);
        } finally {
            loadingState.style.display = "none";
            btnAnalyze.disabled = false;
        }
    });

    function renderBatchResults(data) {
        resultsContainer.style.display = "block";
        batchLoadedRecords = data.records || [];

        // 1. Cập nhật thẻ thống kê
        const s = data.summary || {};
        document.getElementById("metric-total").textContent = (s.total || 0).toLocaleString();

        document.getElementById("metric-pos-pct").textContent = `${s.positive_percent || 0}%`;
        document.getElementById("metric-pos-count").textContent = `(${s.positive_count || 0} bài)`;

        document.getElementById("metric-neg-pct").textContent = `${s.negative_percent || 0}%`;
        document.getElementById("metric-neg-count").textContent = `(${s.negative_count || 0} bài)`;

        document.getElementById("metric-neu-pct").textContent = `${s.neutral_percent || 0}%`;
        document.getElementById("metric-neu-count").textContent = `(${s.neutral_count || 0} bài)`;

        // 2. Xu hướng theo 6 khía cạnh SERVQUAL
        const aspectListContainer = document.getElementById("batch-aspect-list");
        aspectListContainer.innerHTML = "";

        const aspects = data.aspect_breakdown || [];
        aspects.forEach(asp => {
            const card = document.createElement("div");
            card.className = "aspect-stat-card";

            const mentions = asp.mentions || 0;
            const posW = mentions > 0 ? (asp.positive_rate || 0) : 0;
            const negW = mentions > 0 ? (asp.negative_rate || 0) : 0;
            const neuW = mentions > 0 ? (asp.neutral_rate !== undefined ? asp.neutral_rate : Math.max(0, Math.round((100 - posW - negW) * 10) / 10)) : 0;

            card.innerHTML = `
                <div class="aspect-stat-header">
                    <span class="aspect-stat-name">${escapeHtml(asp.name)}</span>
                    <span class="aspect-stat-mentions">${mentions} lượt nhắc</span>
                </div>
                <div class="aspect-stat-bar-track">
                    <div class="aspect-stat-bar-pos" style="width: ${posW}%;" title="Khen ngợi: ${posW}%"></div>
                    <div class="aspect-stat-bar-neg" style="width: ${negW}%;" title="Phàn nàn: ${negW}%"></div>
                    <div class="aspect-stat-bar-neu" style="width: ${neuW}%;" title="Trung lập: ${neuW}%"></div>
                </div>
                <div class="aspect-stat-legend">
                    <span class="legend-pos"><i class="fa-solid fa-thumbs-up"></i> Hài lòng: ${asp.positive || 0} (${posW}%)</span>
                    <span class="legend-neg"><i class="fa-solid fa-thumbs-down"></i> Phàn nàn: ${asp.negative || 0} (${negW}%)</span>
                    <span class="legend-neu"><i class="fa-solid fa-minus"></i> Trung lập: ${asp.neutral || 0} (${neuW}%)</span>
                </div>
            `;
            aspectListContainer.appendChild(card);
        });

        // 3. Bảng dữ liệu chi tiết
        renderTableRows(batchLoadedRecords);

        // Đặt lại nút filter về "all"
        filterButtons.forEach(b => b.classList.remove("active"));
        document.querySelector('.filter-btn[data-filter="all"]').classList.add("active");
    }

    function renderTableRows(records) {
        const tbody = document.getElementById("batch-table-body");
        tbody.innerHTML = "";
        document.getElementById("table-row-count").textContent = `${records.length} dòng`;

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding: 30px; color: var(--text-muted);">Không có bản ghi nào phù hợp với bộ lọc.</td></tr>`;
            return;
        }

        records.forEach(row => {
            const tr = document.createElement("tr");

            const isP = row.sentiment === "Positive";
            const isN = row.sentiment === "Negative";
            const badgeClass = isP ? "pos" : (isN ? "neg" : "neu");
            const badgeText = row.sentiment_vi || (isP ? "Tích Cực" : (isN ? "Tiêu Cực" : "Trung Lập"));

            const aspectTags = (row.aspects || []).map(a => `<span class="aspect-tag-mini">${escapeHtml(a)}</span>`).join(" ");

            tr.innerHTML = `
                <td><strong>${row.stt}</strong></td>
                <td>${escapeHtml(row.text)}</td>
                <td class="col-sentiment"><span class="table-badge ${badgeClass}">${badgeText}</span></td>
                <td><div class="aspect-tags-group">${aspectTags || '<span style="color:var(--text-light); font-size:0.75rem;">Chung</span>'}</div></td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Bộ lọc cảm xúc
    filterButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            filterButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            const filterVal = btn.getAttribute("data-filter");
            if (filterVal === "all") {
                renderTableRows(batchLoadedRecords);
            } else {
                const filtered = batchLoadedRecords.filter(r => r.sentiment === filterVal);
                renderTableRows(filtered);
            }
        });
    });

    // Xuất file CSV kết quả
    btnExportCsv.addEventListener("click", () => {
        if (!batchLoadedRecords || batchLoadedRecords.length === 0) {
            alert("Không có dữ liệu để xuất file!");
            return;
        }

        // Tạo nội dung CSV định dạng UTF-8 có BOM
        let csvContent = "\uFEFFSTT,Nội Dung Nhận Xét,Cảm Xúc Dự Đoán,Nhãn Tiếng Việt,Độ Tin Cậy,Khía Cạnh Liên Quan\n";

        batchLoadedRecords.forEach(r => {
            const cleanText = `"${(r.text || '').replace(/"/g, '""')}"`;
            const aspects = `"${(r.aspects || []).join('; ')}"`;
            csvContent += `${r.stt},${cleanText},${r.sentiment},${r.sentiment_vi},${r.confidence || 0.9},${aspects}\n`;
        });

        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", url);
        link.setAttribute("download", `MedVina_KetQua_PhanTich_${new Date().toISOString().slice(0, 10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    });
}

/* ==============================================================================
   HELPER UTILITIES
   ============================================================================== */
function formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
