const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const mysql = require("mysql2");
const bcrypt = require("bcrypt");

const app = express();
app.use(cors());
app.use(express.json());

const db = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "", 
  database: "mindmate_db"
});

db.connect(err => {
  if (err) {
    console.error("DB connection error:", err);
  } else {
    console.log("✅ Connected to MySQL");
  }
});


app.post("/signup", async (req, res) => {
  const { name, email, password } = req.body;

  try {
    const hashedPassword = await bcrypt.hash(password, 10);

    db.query(
      "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
      [name, email, hashedPassword],
      (err, result) => {
        if (err) {
          return res.status(400).json({ error: "Email already exists" });
        }
        res.json({ success: true });
      }
    );
  } catch (err) {
    res.status(500).json({ error: "Signup failed" });
  }
});

app.post("/login", (req, res) => {
  const { email, password } = req.body;

  db.query(
    "SELECT * FROM users WHERE email = ?",
    [email],
    async (err, results) => {
      if (err) return res.status(500).json({ error: err });

      if (results.length === 0) {
        return res.status(400).json({ error: "User not found" });
      }

      const user = results[0];

      const match = await bcrypt.compare(password, user.password);

      if (!match) {
        return res.status(400).json({ error: "Invalid password" });
      }

      res.json({
        success: true,
        user_id: user.id,
        name: user.name
      });
    }
  );
});

/* ===============================
   PREDICT + SAVE TO DB
================================ */
app.post("/predict", (req, res) => {
  const inputData = req.body;

  const py = spawn("python", ["predict.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", (data) => {
    result += data.toString();
  });

  py.stderr.on("data", (data) => {
    error += data.toString();
  });

  py.on("close", (code) => {
    if (code !== 0) {
      return res.status(500).json({ error });
    }

    try {
      const output = JSON.parse(result);

      // 🔥 SAVE TO DATABASE
      const {
        sleep, work_hours, activity, social, stress_self, user_id
      } = inputData;

      const {
        prediction, confidence,
        academic_pressure, study_satisfaction,
        dietary_habits, financial_stress, depression
      } = output;

      db.query(
        `INSERT INTO checkins 
        (user_id, sleep, work_hours, activity, social, stress_self,
         prediction, confidence,
         academic_pressure, study_satisfaction,
         dietary_habits, financial_stress, depression)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          user_id,
          sleep, work_hours, activity, social, stress_self,
          prediction, confidence,
          academic_pressure, study_satisfaction,
          dietary_habits, financial_stress, depression
        ]
      );

      res.json(output);

    } catch (err) {
      res.status(500).json({ error: "Invalid model output" });
    }
  });
});

/* ===============================
   START SERVER
================================ */
app.listen(3000, () => {
  console.log("🚀 Server running on http://localhost:3000");
});