const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const mysql = require("mysql2");
const bcrypt = require("bcrypt");

const app = express();
app.use(cors());
app.use(express.json());

/* ================= DB CONNECTION (SAFE MODE) ================= */

let db = null;

try {
  db = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "Laasya@123", // ⚠️ change if needed
    database: "mindmate_db"
  });

  db.connect(err => {
    if (err) {
      console.error("❌ DB connection failed (continuing without DB)");
      db = null; // disable DB safely
    } else {
      console.log("✅ Connected to MySQL");
    }
  });

} catch (e) {
  console.error("❌ DB init error:", e);
  db = null;
}

/* ================= SAFE QUERY HELPER ================= */

function safeQuery(sql, params, callback) {
  if (!db) {
    console.warn("⚠️ DB not connected");
    return callback(null, []); // fallback
  }

  db.query(sql, params, (err, results) => {
    if (err) {
      console.error("❌ DB query error:", err);
      return callback(err, null);
    }
    callback(null, results);
  });
}

/* ================= AUTH ================= */

app.post("/signup", async (req, res) => {
  if (!db) return res.json({ success: true, message: "DB offline (mock signup)" });

  const { name, email, password } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    safeQuery(
      "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
      [name, email, hashedPassword],
      (err) => {
        if (err) return res.status(400).json({ error: "Email exists" });
        res.json({ success: true });
      }
    );
  } catch {
    res.status(500).json({ error: "Signup failed" });
  }
});

app.post("/login", (req, res) => {
  if (!db) return res.json({ success: true, user_id: 1, name: "Offline User" });

  const { email, password } = req.body;

  safeQuery(
    "SELECT * FROM users WHERE email = ?",
    [email],
    async (err, results) => {
      if (err) return res.status(500).json({ error: err });
      if (results.length === 0) return res.status(400).json({ error: "User not found" });

      const user = results[0];
      const match = await bcrypt.compare(password, user.password);

      if (!match) return res.status(400).json({ error: "Invalid password" });

      res.json({
        success: true,
        user_id: user.id,
        name: user.name
      });
    }
  );
});

/* ================= PYTHON HELPER ================= */

function runPython(script, inputData, res, errorMsg) {
  const py = spawn("python", [script, JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", d => result += d.toString());
  py.stderr.on("data", d => error += d.toString());

  py.on("close", code => {
    if (code !== 0) {
      console.error(error);
      return res.status(500).json({ error: errorMsg });
    }

    try {
      res.json(JSON.parse(result.trim()));
    } catch {
      console.error("❌ JSON parse error:", result);
      res.status(500).json({ error: "Invalid output" });
    }
  });
}

/* ================= MODULAR ML ================= */

app.post("/predict-depression", (req, res) =>
  runPython("predict_depression.py", req.body, res, "Depression failed")
);

app.post("/predict-financial", (req, res) =>
  runPython("predict_finstress.py", req.body, res, "Financial failed")
);

app.post("/predict-diet", (req, res) =>
  runPython("predict_diet.py", req.body, res, "Diet failed")
);

app.post("/predict-academic", (req, res) =>
  runPython("predict_academic.py", req.body, res, "Academic failed")
);

app.post("/predict-stress", (req, res) =>
  runPython("predict_master.py", req.body, res, "Master failed")
);

/* ================= SAVE ALL ================= */

app.post("/save-all", (req, res) => {
  if (!db) return res.json({ success: true, message: "Saved locally (DB offline)" });

  const {
    user_id,
    sleep,
    activity,
    social,
    work_hours,
    depression,
    financial_stress,
    dietary_habits,
    academic_pressure,
    prediction,
    confidence
  } = req.body;

  const fixedConfidence = confidence <= 1 ? confidence * 100 : confidence;

  const sql = `
    INSERT INTO checkins 
    (user_id, sleep, activity, social, work_hours,
     depression, financial_stress, dietary_habits, academic_pressure,
     prediction, confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `;

  safeQuery(sql, [
    user_id,
    sleep || 0,
    activity || 0,
    social || 0,
    work_hours || 0,
    depression,
    financial_stress,
    dietary_habits,
    academic_pressure,
    prediction,
    fixedConfidence || 50
  ], (err) => {
    if (err) return res.status(500).json({ error: "DB insert failed" });
    res.json({ success: true });
  });
});

/* ================= PROGRESS ================= */

app.get("/progress/:userId", (req, res) => {
  if (!db) return res.json([]);

  const userId = req.params.userId;

  safeQuery(
    `SELECT created_at, confidence FROM checkins WHERE user_id = ? ORDER BY created_at ASC`,
    [userId],
    (err, results) => {
      if (err) return res.status(500).json({ error: "Failed" });

      const formatted = results.map(row => ({
        confidence: row.confidence,
        date: new Date(row.created_at).toLocaleDateString("en-GB", {
          day: "numeric",
          month: "short"
        })
      }));

      res.json(formatted);
    }
  );
});

/* ================= START ================= */

app.listen(3000, () => {
  console.log("🚀 Server running on http://localhost:3000");
});