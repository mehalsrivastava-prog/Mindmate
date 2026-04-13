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
  password: "Shauryas@8092", 
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

app.post("/predict-stress", (req, res) => {
  const inputData = req.body;

  const py = spawn("python", ["predict_stress.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", (data) => result += data.toString());
  py.stderr.on("data", (data) => error += data.toString());

  py.on("close", (code) => {
    if (code !== 0) {
      console.error("Python Error:", error);
      return res.status(500).json({ error: "Stress model failed" });
    }

    try {
      const output = JSON.parse(result);

      const {
        sleep, work_hours, activity, social, stress_self, user_id
      } = inputData;

      const { prediction, confidence } = output;

      db.query(
        `INSERT INTO checkins 
        (user_id, sleep, work_hours, activity, social, stress_self,
         prediction, confidence)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          user_id,
          sleep, work_hours, activity, social, stress_self,
          prediction, confidence
        ],
        (err) => {
          if (err) console.error("Stress DB Error:", err);
        }
      );

      res.json(output);

    } catch (err) {
      console.error("Parse Error:", err);
      res.status(500).json({ error: "Invalid stress model output" });
    }
  });
});


/* ===============================
   DEPRESSION PREDICTION (NEW PAGE)
================================ */
app.post("/predict-depression", (req, res) => {
  const inputData = req.body;

  // ✅ FIXED FILE NAME
  const py = spawn("python", ["predict_depression.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", (data) => result += data.toString());
  py.stderr.on("data", (data) => error += data.toString());

  py.on("close", (code) => {
    if (code !== 0) {
      console.error("Python Error:", error);
      return res.status(500).json({ error: "Depression model failed" });
    }

    try {
      const output = JSON.parse(result);

      const {
        q1, q2, q3, q4, q5, q6, q7, q8, q9,
        user_id
      } = inputData;

      const { prediction, confidence } = output;

      db.query(
        `INSERT INTO depression_checkins 
        (user_id, q1_sad, q2_interest, q3_sleep, q4_energy,
         q5_appetite, q6_guilt, q7_focus, q8_movement, q9_selfharm,
         prediction, confidence)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          user_id,
          q1, q2, q3, q4,
          q5, q6, q7, q8, q9,
          prediction, confidence
        ],
        (err) => {
          if (err) console.error("Depression DB Error:", err);
        }
      );

      res.json(output);

    } catch (err) {
      console.error("Parse Error:", err);
      res.status(500).json({ error: "Invalid depression model output" });
    }
  });
});
app.listen(3000, () => {
  console.log("🚀 Server running on http://localhost:3000");
});

app.post("/predict-financial", (req, res) => {
  const inputData = req.body;

  const py = spawn("python", ["predict_finstress.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", d => result += d.toString());
  py.stderr.on("data", d => error += d.toString());

  py.on("close", code => {
    if (code !== 0) {
      console.error(error);
      return res.status(500).json({ error: "Financial model failed" });
    }

    try {
      res.json(JSON.parse(result));
    } catch {
      res.status(500).json({ error: "Invalid output" });
    }
    

    console.log("Python output:", result);
  });
});
app.post("/predict-diet", (req, res) => {
  const inputData = req.body;

  const py = spawn("python", ["predict_diet.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", (data) => result += data.toString());
  py.stderr.on("data", (data) => error += data.toString());

  py.on("close", (code) => {
  
  if (code !== 0) {
    return res.status(500).json({ error: "Diet model failed" });
  }

  try {
    const output = JSON.parse(result.trim());
    res.json(output);
  } catch (err) {
    console.error("PARSE ERROR:", err);
    res.status(500).json({ error: "Invalid output" });
  }
});
});

app.post("/predict-academic", (req, res) => {
  const inputData = req.body;

  const py = spawn("python", ["predict_academic.py", JSON.stringify(inputData)]);

  let result = "";
  let error = "";

  py.stdout.on("data", (data) => result += data.toString());
  py.stderr.on("data", (data) => error += data.toString());

  py.on("close", (code) => {

    if (code !== 0) {
      console.error("Academic Error:", error);
      return res.status(500).json({ error: "Academic model failed" });
    }

    try {
      const output = JSON.parse(result.trim());
      res.json(output);
    } catch (err) {
      console.error("Parse Error:", err);
      res.status(500).json({ error: "Invalid academic output" });
    }

  });
});