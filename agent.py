"""The submission entrypoint. The platform imports this file and calls get_move.

Negamax with alpha-beta, iterative deepening, a transposition table, MVV-LVA move
ordering, quiescence search on captures, and a material + piece-square evaluation.
"""

import time

import chess
import chess.polyglot

MATE = 10 ** 6
INF = 10 ** 9

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-square tables, white's perspective, a1 = index 0 .. h8 = index 63.
PAWN_PST = [
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10, -20, -20,  10,  10,   5,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,   5,  10,  25,  25,  10,   5,   5,
    10,  10,  20,  30,  30,  20,  10,  10,
    50,  50,  50,  50,  50,  50,  50,  50,
    0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
ROOK_PST = [
    0,   0,   0,   5,   5,   0,   0,   0,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    5,  10,  10,  10,  10,  10,  10,   5,
    0,   0,   0,   0,   0,   0,   0,   0,
]
QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
    0,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
KING_MID_PST = [
    20,  30,  10,   0,   0,  10,  30,  20,
    20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
KING_END_PST = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}


def _mirror(square: int) -> int:
    return square ^ 56


def game_phase(board: chess.Board) -> float:
    """0.0 = opening/middlegame, 1.0 = endgame, based on remaining non-pawn material."""
    total = (
        len(board.pieces(chess.KNIGHT, chess.WHITE)) * 1
        + len(board.pieces(chess.KNIGHT, chess.BLACK)) * 1
        + len(board.pieces(chess.BISHOP, chess.WHITE)) * 1
        + len(board.pieces(chess.BISHOP, chess.BLACK)) * 1
        + len(board.pieces(chess.ROOK, chess.WHITE)) * 2
        + len(board.pieces(chess.ROOK, chess.BLACK)) * 2
        + len(board.pieces(chess.QUEEN, chess.WHITE)) * 4
        + len(board.pieces(chess.QUEEN, chess.BLACK)) * 4
    )
    max_total = 1 * 4 + 1 * 4 + 2 * 4 + 4 * 2  # both sides' starting non-pawn weight
    return 1.0 - min(total, max_total) / max_total


def evaluate(board: chess.Board) -> int:
    """Static evaluation from the perspective of the side to move."""
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = game_phase(board)
    score = 0.0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        idx = square if piece.color == chess.WHITE else _mirror(square)
        if piece.piece_type == chess.KING:
            pst_value = KING_MID_PST[idx] * (1 - phase) + KING_END_PST[idx] * phase
        else:
            pst_value = PST[piece.piece_type][idx]
        total = value + pst_value
        score += total if piece.color == chess.WHITE else -total

    result = round(score) if board.turn == chess.WHITE else -round(score)
    return result


def _mvv_lva_key(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    if victim is None:
        return 0
    attacker_value = PIECE_VALUE.get(attacker, 0) if attacker is not None else 0
    return PIECE_VALUE.get(victim, 0) * 16 - attacker_value


class SearchTimeout(Exception):
    pass


class Engine:
    def __init__(self) -> None:
        self.tt: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        self.deadline = 0.0
        self.nodes = 0

    def _check_time(self) -> None:
        self.nodes += 1
        if self.nodes % 1024 == 0 and time.monotonic() > self.deadline:
            raise SearchTimeout

    def order_moves(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None
    ) -> list[chess.Move]:
        def key(move: chess.Move) -> tuple[int, int]:
            if tt_move is not None and move == tt_move:
                return (3, 0)
            if board.is_capture(move):
                return (2, _mvv_lva_key(board, move))
            if move.promotion:
                return (1, move.promotion)
            return (0, 0)

        return sorted(moves, key=key, reverse=True)

    def quiescence(self, board: chess.Board, alpha: int, beta: int) -> int:
        self._check_time()
        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        moves = self.order_moves(board, moves, None)
        for move in moves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int
    ) -> int:
        self._check_time()

        if board.is_repetition(2) or board.halfmove_clock >= 100:
            return 0

        key = chess.polyglot.zobrist_hash(board)
        tt_move = None
        entry = self.tt.get(key)
        if entry is not None:
            entry_depth, entry_score, flag, entry_move = entry
            tt_move = entry_move
            if entry_depth >= depth:
                if flag == 0:
                    return entry_score
                if flag == 1 and entry_score > alpha:
                    alpha = entry_score
                elif flag == 2 and entry_score < beta:
                    beta = entry_score
                if alpha >= beta:
                    return entry_score

        moves = list(board.legal_moves)
        if not moves:
            if board.is_check():
                return -MATE + (64 - depth)
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        moves = self.order_moves(board, moves, tt_move)

        best_score = -INF
        best_move = None
        orig_alpha = alpha
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
            if alpha >= beta:
                break

        flag = 0
        if best_score <= orig_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        self.tt[key] = (depth, best_score, flag, best_move)

        return best_score

    def search(self, board: chess.Board, time_budget: float) -> chess.Move:
        self.deadline = time.monotonic() + time_budget
        self.tt.clear()
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]

        best_move = legal[0]
        depth = 1
        try:
            while True:
                alpha, beta = -INF, INF
                current_best = None
                current_best_score = -INF
                ordered = self.order_moves(board, legal, best_move)
                for move in ordered:
                    board.push(move)
                    score = -self.negamax(board, depth - 1, -beta, -alpha)
                    board.pop()
                    if score > current_best_score:
                        current_best_score = score
                        current_best = move
                    if current_best_score > alpha:
                        alpha = current_best_score
                if current_best is not None:
                    best_move = current_best
                if current_best_score >= MATE - 64:
                    break
                depth += 1
                if depth > 64:
                    break
        except SearchTimeout:
            pass

        return best_move


_engine = Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion
    """
    board = chess.Board(fen)

    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        return legal[0].uci()

    # Budget: assume ~30 moves left in a typical game, keep a safety margin, never
    # burn more than a third of the remaining clock on one move, and always leave
    # room for the per-move increment plus overhead.
    time_left_s = max(time_left_ms, 0) / 1000.0
    expected_moves_left = 40
    budget = time_left_s / expected_moves_left
    budget = min(budget, time_left_s / 3.0)
    budget = max(budget, 0.05)
    budget -= 0.05  # safety margin for overhead outside the search loop
    budget = max(budget, 0.02)

    move = _engine.search(board, budget)
    return move.uci()
