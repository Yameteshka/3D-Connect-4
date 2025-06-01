<h1 align="center">🟦🟨 3D Connect-4 AI (5×5×5) 🟥🟩</h1>

<p align="center">
  <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExOGwyemZueXB6ZDU1aXRsbnB2ZGtvdDN6Y3A2cjVjNnBoNThlanBrMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/GqDnImZ3UdJhS/giphy.gif" width="300" />
</p>

<p align="center">
  <b>Modern, fast and explainable AI algorithm for 3D Connect-4 on a 5×5×5 board.<br>
  Pure Python + Numba-accelerated minimax + hand-tuned evaluation.<br>
  Play vs the AI — experience multi-dimensional strategy!</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-completed-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/license-MIT-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/author-Yameteshka-ff69b4?style=for-the-badge" />
</p>

---

## 🕹️ Game Rules

- The board is a 5×5×5 cube (3D grid).
- Two players (you vs AI) take turns placing pieces in any column.
- **Gravity:** Pieces always fall to the lowest empty cell in their (x, y) column.
- The goal: **get four in a row** — horizontally, vertically, or diagonally, in any dimension (including 3D diagonals!).
- First to connect four of their pieces wins. If the board is full, it's a draw.

---

## 🚀 Getting Started

> **For the best experience, run `game_interface.py`** — it includes an intuitive user interface and sounds.

```bash
pip install numpy numba
python game_interface.py
```

Or for CLI-only mode:

```bash
python connect4_3d.py
```

---

## ✨ Example Gameplay

<p align="center">
  <img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExOXE4Z2ZhYzRqdDE0bDNjb3NidjdhZGczZDRnMDJidm1lam54MWRpbiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/g1WeceyrXG4RWsA7zX/giphy.gif" width="390" />
</p>

---

## 🧠 How It Works

- **Board:** 5×5×5 numpy array (`0 = empty, 1 = player, -1 = AI`)
- **Minimax:** Adaptive search depth with alpha-beta pruning, fast move ordering, and transposition table
- **Threat Analysis:** Detects forced wins, blocks, and open threats
- **Numba:** Win checking and evaluation run at near-C speed for smooth gameplay
- **Gravity:** Pieces drop to the bottom of each column

---

## 🎮 Features

- **3D strategy:** Play against a challenging AI in true 3D
- **Interactive interface:** Visuals and sound in `game_interface.py`
- **CLI fallback:** Classic console gameplay in `connect4_3d.py`
- **Clean, documented code:** Easy to read, extend, and reuse

---

## 🛠️ Future Plans

- **Roblox version coming soon!**  
  Porting this AI engine to Roblox to support multiplayer online play, smooth 3D graphics, and intuitive controls.

---

## 📜 MIT License

```
MIT License

Copyright (c) 2025 Julia Kurnaeva (@Yameteshka)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND.
```

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer"/>
</p>
