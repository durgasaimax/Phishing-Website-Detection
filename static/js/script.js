
let currentScan = null;


// ========================================
// SCAN WEBSITE
// ========================================

async function scanWebsite() {

    const urlInput = document.getElementById("urlInput");
    const scanButton = document.getElementById("scanButton");

    const url = urlInput.value.trim();

    const loading = document.getElementById("loading");
    const errorMessage = document.getElementById("errorMessage");
    const resultSection = document.getElementById("resultSection");

    errorMessage.classList.add("hidden");
    resultSection.classList.add("hidden");

    if (!url) {
        showError("Please enter a website URL.");
        return;
    }

    scanButton.disabled = true;
    loading.classList.remove("hidden");

    try {

        const response = await fetch("/scan", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Unable to scan URL.");
        }

        currentScan = {
            url: url,
            ...data
        };

        displayResult(data);

        loadHistory();

    } catch (error) {

        showError(error.message);

    } finally {

        loading.classList.add("hidden");
        scanButton.disabled = false;

    }
}


// ========================================
// DISPLAY RESULT
// ========================================

function displayResult(data) {

    const resultSection = document.getElementById("resultSection");

    const resultCard = document.getElementById("resultCard");

    const resultIcon = document.getElementById("resultIcon");

    const resultTitle = document.getElementById("resultTitle");

    const resultDescription = document.getElementById("resultDescription");

    const probabilityValue = document.getElementById("probabilityValue");

    const probabilityBar = document.getElementById("probabilityBar");

    const probability = Number(data.phishing_probability);

    resultSection.classList.remove("hidden");

    if (data.prediction === 0) {

        resultCard.className = "result-card danger";

        resultIcon.textContent = "⚠";

        resultTitle.textContent = "PHISHING DETECTED";

        resultDescription.textContent =
            "The model classified this URL as potentially malicious. Avoid entering sensitive information.";

    } else {

        resultCard.className = "result-card safe";

        resultIcon.textContent = "✓";

        resultTitle.textContent = "LEGITIMATE CLASSIFICATION";

        resultDescription.textContent =
            "The model classified this URL as legitimate. This does not guarantee that the website is safe.";

    }

    probabilityValue.textContent = probability.toFixed(2) + "%";

    probabilityBar.style.width =
        Math.min(100, Math.max(0, probability)) + "%";


    // URL intelligence

    const features = data.features;

    document.getElementById("urlLength").textContent =
        features.url_length;

    document.getElementById("domainLength").textContent =
        features.domain_length;

    document.getElementById("suspiciousWords").textContent =
        features.suspicious_word_count;

    document.getElementById("httpsStatus").textContent =
        features.is_https ? "Enabled" : "Disabled";

    document.getElementById("ipStatus").textContent =
        features.has_ip_address ? "Detected" : "Not detected";

    document.getElementById("subdomains").textContent =
        features.subdomain_count_approx;

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


// ========================================
// SHOW ERROR
// ========================================

function showError(message) {

    const errorMessage = document.getElementById("errorMessage");

    errorMessage.textContent = message;

    errorMessage.classList.remove("hidden");

}


// ========================================
// LOAD SCAN HISTORY
// ========================================

async function loadHistory() {

    try {

        const response = await fetch("/history");

        if (!response.ok) {
            throw new Error("Unable to load history.");
        }

        const history = await response.json();

        const historyBody = document.getElementById("historyBody");

        historyBody.replaceChildren();

        if (history.length === 0) {

            const row = document.createElement("tr");
            const cell = document.createElement("td");

            cell.colSpan = 4;
            cell.className = "empty-state";
            cell.textContent = "No scans available yet.";

            row.appendChild(cell);
            historyBody.appendChild(row);

            return;
        }

        history.slice().reverse().forEach(item => {

            const row = document.createElement("tr");

            const dateCell = document.createElement("td");
            dateCell.textContent = item.date;

            const urlCell = document.createElement("td");
            urlCell.textContent = item.url;

            const resultCell = document.createElement("td");
            resultCell.textContent = item.result;

            resultCell.className =
                item.result === "Phishing"
                    ? "status-danger"
                    : "status-safe";

            const probabilityCell = document.createElement("td");

            probabilityCell.textContent =
                Number(item.phishing_probability).toFixed(2) + "%";

            row.append(
                dateCell,
                urlCell,
                resultCell,
                probabilityCell
            );

            historyBody.appendChild(row);

        });

    } catch (error) {

        console.error(error);

    }

}


// ========================================
// CLEAR HISTORY
// ========================================

async function clearHistory() {

    if (!confirm("Are you sure you want to clear scan history?")) {
        return;
    }

    try {

        const response = await fetch("/clear-history", {
            method: "POST"
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error("Unable to clear history.");
        }

        currentScan = null;

        loadHistory();

    } catch (error) {

        alert(error.message);

    }

}


// ========================================
// DOWNLOAD CURRENT SCAN REPORT
// ========================================

function downloadCurrentReport() {

    if (!currentScan) {
        alert("Please scan a website first.");
        return;
    }

    const data = currentScan;

    const report = {
        Date: new Date().toISOString(),
        URL: data.url,
        Prediction: data.result,
        "Phishing Probability (%)": data.phishing_probability,
        ...data.features
    };

    const csv = convertToCSV([report]);

    downloadCSV(csv, "phishguard_scan_report.csv");

}


// ========================================
// DOWNLOAD HISTORY
// ========================================

function downloadHistory() {

    window.location.href = "/download-report";

}


// ========================================
// CSV UTILITIES
// ========================================

function convertToCSV(data) {

    if (!data.length) {
        return "";
    }

    const headers = Object.keys(data[0]);

    const rows = data.map(row => {

        return headers.map(header => {

            const value = String(row[header] ?? "");

            return '"' + value.replace(/"/g, '""') + '"';

        }).join(",");

    });

    return [
        headers.join(","),
        ...rows
    ].join("\n");

}


function downloadCSV(csv, filename) {

    const blob = new Blob(
        [csv],
        { type: "text/csv;charset=utf-8;" }
    );

    const link = document.createElement("a");

    const url = URL.createObjectURL(blob);

    link.href = url;
    link.download = filename;

    document.body.appendChild(link);

    link.click();

    link.remove();

    URL.revokeObjectURL(url);

}


// ========================================
// NAVIGATION ACTIVE STATE
// ========================================

document.querySelectorAll(".nav-link").forEach(link => {

    link.addEventListener("click", function() {

        document.querySelectorAll(".nav-link").forEach(nav => {
            nav.classList.remove("active");
        });

        this.classList.add("active");

    });

});


// ========================================
// ENTER KEY TO SCAN
// ========================================

document.getElementById("urlInput").addEventListener("keydown", function(event) {

    if (event.key === "Enter") {
        scanWebsite();
    }

});


// ========================================
// INITIALIZE WEBSITE
// ========================================

document.addEventListener("DOMContentLoaded", function() {

    loadHistory();

});
