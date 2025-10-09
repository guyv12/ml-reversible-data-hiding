# Project Setup Instructions

## 1. Create a Virtual Environment

```bash
python -m venv .venv
# or
python3 -m venv .venv
```

---

## 2. Activate the Virtual Environment

* **Linux / macOS:**

```bash
source .venv/bin/activate
```

* **Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

* **Windows (CMD):**

```cmd
.venv\Scripts\activate.bat
```

---

## 3. Upgrade tools (recommended)

```bash
pip install --upgrade pip
# and 
pip install --upgrade build
```
---

## 4. Install Your Package in Editable Mode

```bash
pip install -e .
```
