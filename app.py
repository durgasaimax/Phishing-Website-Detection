
from flask import Flask, render_template, request, jsonify
from inference import predict_website
from datetime import datetime
import csv
import io

app = Flask(__name__)

# Store scan history during the current server session
scan_history = []


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan():

    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({
            "success": False,
            "error": "Please enter a website URL."
        }), 400

    try:
        result = predict_website(url)

        scan_record = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": result["url"],
            "result": result["result"],
            "phishing_probability": round(
                result["phishing_probability"] * 100, 2
            )
        }

        scan_history.append(scan_record)

        return jsonify({
            "success": True,
            "result": result["result"],
            "prediction": result["prediction"],
            "phishing_probability": round(
                result["phishing_probability"] * 100, 2
            ),
            "features": result["features"]
        })

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception as error:
        app.logger.exception("URL prediction failed")
        return jsonify({
            "success": False,
            "error": "Unable to analyse this URL. Please try again."
        }), 500


@app.route("/history")
def history():
    return jsonify(scan_history)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    scan_history.clear()
    return jsonify({"success": True})


@app.route("/download-report")
def download_report():

    output = io.StringIO()

    fieldnames = [
        "date",
        "url",
        "result",
        "phishing_probability"
    ]

    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(scan_history)

    from flask import Response

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=phishguard_scan_history.csv"
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
