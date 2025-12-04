# app.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Modell und Label-Encoder laden
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class_names = list(label_encoder.classes_)

# Preprocessing / OneHotEncoder holen
preprocess = model.named_steps["preprocess"]
ohe = preprocess.named_transformers_["cat"]

# Kategorien:
job_titles = list(ohe.categories_[0])        # bekannte Jobtitel
education_levels = list(ohe.categories_[1])  # bekannte Education-Levels


@app.route("/")
def index():
    # Dropdown-Optionen für bekannte Jobtitel
    job_title_options = "".join(
        f'<option value="{jt}">{jt}</option>' for jt in job_titles
    )

    # Dropdown-Optionen für Bildung
    edu_options = "".join(
        f'<option value="{ed}">{ed}</option>' for ed in education_levels
    )

    return f"""
    <!doctype html>
    <html lang="de">
    <head>
      <meta charset="utf-8">
      <title>Wird mein Job durch KI ersetzt?</title>
      <style>
        body {{ font-family: sans-serif; max-width: 700px; margin: 40px auto; }}
        label {{ display: block; margin-top: 10px; }}
        input, select {{ width: 100%; padding: 6px; }}
        button {{ margin-top: 15px; padding: 8px 12px; cursor: pointer; }}
        #result {{ margin-top: 20px; font-weight: bold; white-space: pre-line; }}
        .note {{
            margin-top: 20px;
            padding: 10px;
            background: #f0f0f0;
            border-left: 4px solid #777;
            font-size: 0.9em;
        }}
        .error {{
            color: #b00020;
            margin-top: 10px;
            font-size: 0.9em;
        }}
      </style>
    </head>
    <body>
      <h1>Wird mein Job durch KI ersetzt?</h1>
      <p>Gib ein, was du über deinen Job weißt. Das Modell sagt dir, wie hoch das Risiko ist.</p>

      <label>Job-Titel (aus Liste wählen):
        <select id="job_title_select">
          <option value="">-- Bitte wählen --</option>
          {job_title_options}
          <option value="__other__">Anderer Job (siehe Feld unten)</option>
        </select>
      </label>

      <label id="other_job_wrapper" style="display:none;">
        Anderer Jobtitel (frei eingeben):
        <input id="job_title_other" type="text" placeholder="z.B. Game Designer">
      </label>

      <label>Höchster Bildungsabschluss:
        <select id="education_level">
          {edu_options}
        </select>
      </label>

      <label>Berufserfahrung (Jahre):
        <input id="years_experience" type="number" step="0.1" value="5">
      </label>

      <label>Bruttojahresgehalt in USD (Schätzung reicht):
        <input id="average_salary" type="number" step="1000" value="50000">
      </label>

      <div class="note">
        <strong>Hinweis:</strong><br>
        Wenn du einen Jobtitel eingibst, der im Trainingsdatensatz nicht vorkam,
        kann das Modell ihn trotzdem verarbeiten. In diesem Fall basiert die Einschätzung
        überwiegend auf deinem Bildungsabschluss, deiner Berufserfahrung und deinem Gehalt.
      </div>

      <button onclick="sendPredict()">Vorhersage anzeigen</button>

      <div id="result"></div>
      <div id="error" class="error"></div>

      <script>
        // TEXTFELD für "Anderer Job" anzeigen/verstecken
        document.addEventListener("DOMContentLoaded", () => {{
            const select = document.getElementById("job_title_select");
            const otherWrapper = document.getElementById("other_job_wrapper");

            select.addEventListener("change", () => {{
                if (select.value === "__other__") {{
                    otherWrapper.style.display = "block";
                }} else {{
                    otherWrapper.style.display = "none";
                    document.getElementById("job_title_other").value = "";
                }}
            }});
        }});

        async function sendPredict() {{
          const sel = document.getElementById('job_title_select').value;
          const other = document.getElementById('job_title_other').value.trim();
          const edu = document.getElementById('education_level').value;
          const years_experience = parseFloat(document.getElementById('years_experience').value);
          const average_salary = parseFloat(document.getElementById('average_salary').value);

          const errDiv = document.getElementById('error');
          const resDiv = document.getElementById('result');
          errDiv.textContent = "";
          resDiv.textContent = "";

          // Logik zur Jobtitelauswahl
          let job_title = null;
          if (sel === "__other__" && other) {{
            job_title = other;
          }} else if (sel && sel !== "__other__") {{
            job_title = sel;
          }} else if (!sel && other) {{
            job_title = other;
          }} else {{
            errDiv.textContent = "Bitte wähle einen Job aus der Liste oder gib einen eigenen Jobtitel ein.";
            return;
          }}

          if (!edu) {{
            errDiv.textContent = "Bitte wähle einen Bildungsabschluss aus.";
            return;
          }}

          if (isNaN(years_experience) || isNaN(average_salary)) {{
            errDiv.textContent = "Bitte gültige Zahlen für Erfahrung und Gehalt eingeben.";
            return;
          }}

          const payload = {{
            job_title: job_title,
            education_level: edu,
            years_experience: years_experience,
            average_salary: average_salary
          }};

          resDiv.textContent = "Bitte warten...";

          try {{
            const response = await fetch('/predict', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(payload)
            }});

            if (!response.ok) {{
              const errText = await response.text();
              resDiv.textContent = "";
              errDiv.textContent = "Fehler: " + errText;
              return;
            }}

            const data = await response.json();

            const cls = data.class_label;
            const pred = data.prediction;
            const probs = data.probabilities;
            const classNames = data.class_names;

            let probText = "Wahrscheinlichkeiten:\\n";
            for (let i = 0; i < probs.length; i++) {{
              probText += "  " + classNames[i] + ": " + probs[i].toFixed(3) + "\\n";
            }}

            let replaceText = "";
            const clsLower = cls.toLowerCase();
            if (clsLower.includes("high")) {{
              replaceText = "\\nEinschätzung: Dein Job ist stark gefährdet (hohes Risiko).";
            }} else if (clsLower.includes("medium")) {{
              replaceText = "\\nEinschätzung: Es gibt ein mittleres Risiko, dass Teile deines Jobs ersetzt werden.";
            }} else {{
              replaceText = "\\nEinschätzung: Dein Job ist eher sicher (niedriges Risiko).";
            }}

            resDiv.textContent =
              "Vorhergesagte Risiko-Klasse: " + cls + " (Index: " + pred + ")\\n" +
              probText + replaceText;

          }} catch (err) {{
            resDiv.textContent = "";
            errDiv.textContent = "Fehler beim Request: " + err;
          }}
        }}
      </script>
    </body>
    </html>
    """


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        job_title = data["job_title"]
        education_level = data["education_level"]
        years_experience = float(data["years_experience"])
        average_salary = float(data["average_salary"])
    except:
        return jsonify({"error": "Ungültige oder fehlende Eingaben"}), 400

    row = pd.DataFrame([{
        "Job_Title": job_title,
        "Education_Level": education_level,
        "Years_Experience": years_experience,
        "Average_Salary": average_salary,
    }])

    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0].tolist()
    class_label = label_encoder.inverse_transform([pred])[0]

    return jsonify({
        "prediction": pred,
        "class_label": class_label,
        "probabilities": proba,
        "class_names": class_names
    })


if __name__ == "__main__":
    app.run(debug=True)
