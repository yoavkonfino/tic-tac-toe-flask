import os
import random

from flask import Flask, jsonify, request, url_for
from markupsafe import escape
from Agent import Agent

app = Flask(__name__)


# ======================================================================
# Pages (all of them share static/style.css)
# ======================================================================

def page(title, body):
    # wraps a page body in the shared card layout
    return (
        '<!doctype html><html lang="en"><head>'
        '<meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>" + title + "</title>"
        '<link rel="stylesheet" href="' + url_for("static", filename="style.css") + '">'
        '</head><body><div class="card">' + body + "</div></body></html>"
    )


@app.route("/")
def home():
    return page("Home", (
        "<h1>Hello, World!</h1>"
        '<form action="/greet" class="controls">'
        '<input name="name" placeholder="Your name">'
        "<button>Say hello</button>"
        "</form>"
        '<a class="button primary" href="/tictactoe">Play Tic-Tac-Toe</a>'
    ))


@app.route("/hello/<name>")
def hello(name):
    name = escape(name)
    return page("Welcome", (
        f"<h1>Hey, {name}.</h1>"
        '<p class="muted">Good to have you here.</p>'
        '<a href="/">Back home</a>'
    ))


@app.route("/about")
def about():
    return page("About Me", (
        "<h1>About Me</h1>"
        '<p class="muted">Hi, I\'m Yoav! I\'m a student from Israel who enjoys technology, '
        "programming, and learning new things. I'm currently exploring web development "
        "with Python and Flask.</p>"
        '<a href="/">Back home</a>'
    ))


@app.route("/greet")
def greet():
    name = escape(request.args.get("name", "stranger"))
    return page("Welcome", (
        f"<h1>Hey, {name}.</h1>"
        '<p class="muted">Good to have you here.</p>'
        '<a href="/">Back home</a>'
    ))


# ======================================================================
# Tic-tac-toe page (the file lives in static/tictactoe.html)
# ======================================================================

@app.route("/tictactoe")
def tictactoe():
    return app.send_static_file("tictactoe.html")


# ======================================================================
# Tic-tac-toe API: POST /api/move
# The first player is always X and the second is O. Either the human or the
# agent can start. Who is who is inferred from the piece counts on the board
# the agent receives (see roles()).
# Board is a 9-character string of "X", "O", "-" (index 0-8, row by row).
# ======================================================================

EMPTY = "-"

# Each line is in ascending order, as the contract requires for winning_line.
LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
    [0, 4, 8], [2, 4, 6],             # diagonals
]


def find_winning_line(board, player):
    # the first line fully owned by player, else None
    for line in LINES:
        if all(board[i] == player for i in line):
            return line
    return None


def random_agent(board):
    # pick a random empty cell
    return random.choice([i for i, c in enumerate(board) if c == EMPTY])


# the trained Q-table sits next to app.py
Q_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")

smart = Agent(epsilon=0.0)
try:
    smart.load(Q_TABLE_PATH)
except OSError:
    smart = None
    print("q_table.json not found: the qtable agent is disabled")


def qtable_agent(board):
    # the Q-table uses "." for empty cells, the API uses "-"
    state = board.replace(EMPTY, ".")
    legal = [i for i, c in enumerate(state) if c == "."]
    row = smart.q_table.get(state)
    if row is None:
        # a position the Q-table has never seen: play randomly
        return random.choice(legal)
    best = max(row[i] for i in legal)
    return random.choice([i for i in legal if row[i] == best])


# to add an agent: write a function board -> cell index and register it here
AGENTS = {"random": random_agent}
if smart is not None:
    AGENTS["qtable"] = qtable_agent


def api_error(message, code=400):
    return jsonify({"error": message}), code


def roles(board):
    # returns (human_symbol, agent_symbol), worked out from the piece counts
    # equal counts: X is to move, so the agent is X (the agent started)
    # X = O + 1: O is to move, so the agent is O (the human started)
    if board.count("X") == board.count("O"):
        return "O", "X"
    return "X", "O"


def validate_board(board):
    # returns an error message, or None if the board is valid
    if (
        not isinstance(board, str)
        or len(board) != 9
        or any(c not in ("X", "O", EMPTY) for c in board)
    ):
        return "invalid board: must be 9 characters from X, O, -"

    if board.count("X") - board.count("O") not in (0, 1):
        return "invalid board: impossible position"

    # The agent has not just moved, so it cannot already have a winning line.
    _, agent_symbol = roles(board)
    if find_winning_line(board, agent_symbol):
        return "invalid board: impossible position"

    return None


@app.route("/api/move", methods=["POST"])
def move():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return api_error("request body must be a JSON object")

    if "board" not in data:
        return api_error("missing field: board")
    if "agent" not in data:
        return api_error("missing field: agent")

    board = data["board"]
    agent = data["agent"]

    problem = validate_board(board)
    if problem:
        return api_error(problem)

    if not isinstance(agent, str) or agent not in AGENTS:
        return api_error(f"unknown agent: {agent}")

    human_symbol, agent_symbol = roles(board)

    # Did the human's move end the game? Then the agent does not move.
    line = find_winning_line(board, human_symbol)
    if line:
        return jsonify({
            "board": board,
            "agent_move": None,
            "status": "human_wins",
            "winning_line": line,
        })
    if EMPTY not in board:
        return jsonify({
            "board": board,
            "agent_move": None,
            "status": "draw",
            "winning_line": None,
        })

    # Agent replies.
    agent_move = AGENTS[agent](board)
    new_board = board[:agent_move] + agent_symbol + board[agent_move + 1:]

    line = find_winning_line(new_board, agent_symbol)
    if line:
        status = "agent_wins"
    elif EMPTY not in new_board:
        status = "draw"
    else:
        status = "playing"

    return jsonify({
        "board": new_board,
        "agent_move": agent_move,
        "status": status,
        "winning_line": line,
    })


@app.errorhandler(405)
def method_not_allowed(e):
    # JSON error for the API only; other routes keep Flask's normal page.
    if request.path.startswith("/api/"):
        return api_error("method not allowed", 405)
    return e


if __name__ == "__main__":
    app.run(debug=True, port=5001)