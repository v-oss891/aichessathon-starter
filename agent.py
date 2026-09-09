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


def _chebyshev(a: int, b: int) -> int:
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b)),
    )


def _center_distance(square: int) -> int:
    """Chebyshev distance from the nearest of the four center squares (0..3)."""
    file_dist = min(abs(chess.square_file(square) - 3), abs(chess.square_file(square) - 4))
    rank_dist = min(abs(chess.square_rank(square) - 3), abs(chess.square_rank(square) - 4))
    return max(file_dist, rank_dist)


MOPUP_MATERIAL_THRESHOLD = 500
MOPUP_LOSER_MAX_MATERIAL = 500
MOPUP_MIN_PHASE = 0.80
MOPUP_EDGE_WEIGHT = 10
MOPUP_KING_WEIGHT = 4

# Small, standard positional bonuses/penalties on top of material+PST. Each is
# well-established in chess literature and small enough to nudge close decisions
# without overriding the core evaluation.
BISHOP_PAIR_BONUS = 40
ROOK_ON_OPEN_FILE_BONUS = 20
ROOK_ON_SEMI_OPEN_FILE_BONUS = 10
DOUBLED_PAWN_PENALTY = 15

# Passed pawn bonus, indexed by how far the pawn has advanced from its own back
# rank (so index 6 is one square from promoting). Deliberately steep at the top:
# material + the pawn PST together value a pawn on the seventh at ~150cp, but a
# passer one move from queening is worth most of a queen, and the search cannot
# be relied on to discover that -- the tactical justification is often ten plies
# out, well past the horizon in a middlegame. Without this the engine happily
# trades into positions with enemy pawns sitting on the second rank.
PASSED_PAWN_BONUS = (0, 5, 15, 30, 60, 110, 200, 0)


def _build_passed_pawn_masks() -> tuple[list[int], list[int]]:
    """For each square, the squares an enemy pawn must occupy to stop a passer.

    A pawn is passed when no enemy pawn stands on its own file or either
    adjacent file anywhere ahead of it. Precomputed once at import, off the
    clock, so the check at eval time is a single bitboard AND.
    """
    white_masks = [0] * 64
    black_masks = [0] * 64
    for square in range(64):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        white_mask = 0
        black_mask = 0
        for neighbour in (file_index - 1, file_index, file_index + 1):
            if not 0 <= neighbour <= 7:
                continue
            for ahead in range(rank_index + 1, 8):
                white_mask |= chess.BB_SQUARES[chess.square(neighbour, ahead)]
            for behind in range(0, rank_index):
                black_mask |= chess.BB_SQUARES[chess.square(neighbour, behind)]
        white_masks[square] = white_mask
        black_masks[square] = black_mask
    return white_masks, black_masks


PASSED_MASK_WHITE, PASSED_MASK_BLACK = _build_passed_pawn_masks()


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
    white_material = 0
    black_material = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUE[piece.piece_type]
        if piece.piece_type != chess.KING:
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
        idx = square if piece.color == chess.WHITE else _mirror(square)
        if piece.piece_type == chess.KING:
            pst_value = KING_MID_PST[idx] * (1 - phase) + KING_END_PST[idx] * phase
        else:
            pst_value = PST[piece.piece_type][idx]
        total = value + pst_value
        score += total if piece.color == chess.WHITE else -total

    # Mop-up: material + PST alone can't tell a shuffle apart from real progress
    # once one side has an overwhelming, decided advantage — every legal move
    # scores the same, so the search has no reason to actually finish the game
    # and can drift into repeating itself right past the win. When one side is
    # up more than a rook, add a small, separate incentive to drive the losing
    # king to the edge and bring the winning king in, which is enough to break
    # the tie between "shuffle" and "make progress" without disturbing normal
    # play anywhere the game is still genuinely contested.
    material_diff = white_material - black_material
    loser_material = min(white_material, black_material)
    decisive = abs(material_diff) >= MOPUP_MATERIAL_THRESHOLD
    truly_simplified = phase >= MOPUP_MIN_PHASE
    if truly_simplified and loser_material <= MOPUP_LOSER_MAX_MATERIAL and decisive:
        winner = chess.WHITE if material_diff > 0 else chess.BLACK
        winner_king = board.king(winner)
        loser_king = board.king(not winner)
        if winner_king is not None and loser_king is not None:
            mopup = MOPUP_EDGE_WEIGHT * _center_distance(loser_king) + MOPUP_KING_WEIGHT * (
                7 - _chebyshev(winner_king, loser_king)
            )
            score += mopup if winner == chess.WHITE else -mopup

    # Bishop pair: two bishops covering both color complexes are worth more than
    # the sum of their piece values -- standard ~+30-50cp bonus in every serious
    # chess engine.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS

    # Doubled pawns and rook activity by file. One pass through the eight files
    # covers both: count pawns per file per side (>=2 = penalty), then note which
    # files have zero pawns (open) or only enemy pawns (semi-open from our side)
    # to score rooks sitting on them.
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    white_rooks = board.pieces(chess.ROOK, chess.WHITE)
    black_rooks = board.pieces(chess.ROOK, chess.BLACK)
    for file_index in range(8):
        file_mask = chess.BB_FILES[file_index]
        white_file_pawns = bin(int(white_pawns) & file_mask).count("1")
        black_file_pawns = bin(int(black_pawns) & file_mask).count("1")
        if white_file_pawns >= 2:
            score -= DOUBLED_PAWN_PENALTY * (white_file_pawns - 1)
        if black_file_pawns >= 2:
            score += DOUBLED_PAWN_PENALTY * (black_file_pawns - 1)
        white_rooks_on_file = bin(int(white_rooks) & file_mask).count("1")
        black_rooks_on_file = bin(int(black_rooks) & file_mask).count("1")
        if white_rooks_on_file:
            if white_file_pawns == 0 and black_file_pawns == 0:
                score += ROOK_ON_OPEN_FILE_BONUS * white_rooks_on_file
            elif white_file_pawns == 0:
                score += ROOK_ON_SEMI_OPEN_FILE_BONUS * white_rooks_on_file
        if black_rooks_on_file:
            if white_file_pawns == 0 and black_file_pawns == 0:
                score -= ROOK_ON_OPEN_FILE_BONUS * black_rooks_on_file
            elif black_file_pawns == 0:
                score -= ROOK_ON_SEMI_OPEN_FILE_BONUS * black_rooks_on_file

    # Passed pawns. Material and PST alone price a pawn on the seventh at about
    # 150cp, when an unopposed one is worth most of a queen -- the engine will
    # otherwise trade into a lost ending because the refutation sits past its
    # horizon.
    black_pawn_bb = int(black_pawns)
    white_pawn_bb = int(white_pawns)
    for square in white_pawns:
        if not black_pawn_bb & PASSED_MASK_WHITE[square]:
            score += PASSED_PAWN_BONUS[chess.square_rank(square)]
    for square in black_pawns:
        if not white_pawn_bb & PASSED_MASK_BLACK[square]:
            score -= PASSED_PAWN_BONUS[7 - chess.square_rank(square)]

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


MAX_TT_ENTRIES = 2_000_000
# Kept low deliberately: a higher cap lets a check-heavy line accumulate far more
# real search depth than a quiet sibling move at the same nominal depth, which
# inflates its score simply from being explored more thoroughly rather than
# because it's actually better -- confirmed directly by comparing a checking
# move against a quiet alternative at increasing depth: at cap 12 the checking
# move's score kept climbing away from the quiet move's (494 vs 479), and at
# cap 4 they converged to nearly equal (480 vs 479), which is what two genuinely
# comparable moves should look like.
MAX_CHECK_EXTENSIONS = 4
NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2


class Engine:
    def __init__(self) -> None:
        # Kept for the whole game: the platform starts one process per game, so a
        # position searched on an earlier move is still valid to reuse on this one.
        self.tt: dict[int, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: dict[int, list[chess.Move]] = {}
        self.history: dict[tuple[bool, int, int], int] = {}
        # Real occurrences of each position across the actual game so far, recorded
        # once per get_move call. A fresh board() from the fen has no move history of
        # its own, so without this the search cannot see a repetition coming from
        # earlier moves and can walk straight into a threefold draw it never meant to
        # take, or fail to force one when behind.
        self.history_counts: dict[int, int] = {}
        self.path_counts: dict[int, int] = {}
        self.deadline = 0.0
        self.nodes = 0

    def _check_time(self) -> None:
        self.nodes += 1
        if self.nodes % 1024 == 0 and time.monotonic() > self.deadline:
            raise SearchTimeout

    def order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        tt_move: chess.Move | None,
        depth: int = 0,
    ) -> list[chess.Move]:
        killers = self.killers.get(depth, ())

        def key(move: chess.Move) -> tuple[int, int]:
            if tt_move is not None and move == tt_move:
                return (4, 0)
            if board.is_capture(move):
                return (3, _mvv_lva_key(board, move))
            if move.promotion:
                return (2, move.promotion)
            if move in killers:
                return (1, 0)
            return (0, self.history.get((board.turn, move.from_square, move.to_square), 0))

        return sorted(moves, key=key, reverse=True)

    def _record_killer(self, depth: int, move: chess.Move) -> None:
        slot = self.killers.setdefault(depth, [])
        if move in slot:
            return
        slot.insert(0, move)
        del slot[2:]

    def _record_history(self, board: chess.Board, move: chess.Move, depth: int) -> None:
        key = (board.turn, move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

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
            try:
                score = -self.quiescence(board, -beta, -alpha)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, extensions: int = 0
    ) -> int:
        self._check_time()

        if board.halfmove_clock >= 100:
            return 0

        key = chess.polyglot.zobrist_hash(board)

        # Would landing here be this position's real third occurrence in the game
        # (counting moves already played plus this hypothetical continuation)? If so
        # the referee draws it regardless of what the static eval thinks, so score it
        # as the forced draw it is rather than searching past it.
        path_seen = self.path_counts.get(key, 0)
        if self.history_counts.get(key, 0) + path_seen + 1 >= 3:
            return 0

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

        # Null-move pruning: let the side to move pass and search the rest at a
        # reduced depth. If the position is still at least as good for them as beta
        # even after giving the opponent a free move, a real move only does better,
        # so this branch can't be part of the principal line — prune it. Skipped in
        # check (a null move there is illegal), near mate scores (the reduced search
        # isn't reliable that close to forced lines), and in king-and-pawn endings
        # (a free move can be the only thing preventing zugzwang there, so the
        # assumption a free move can only help the giver breaks down).
        if (
            depth >= NULL_MOVE_MIN_DEPTH
            and not board.is_check()
            and abs(beta) < MATE - 100
            and any(
                board.pieces(pt, board.turn)
                for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
            )
        ):
            board.push(chess.Move.null())
            try:
                null_score = -self.negamax(
                    board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, extensions
                )
            finally:
                board.pop()
            if null_score >= beta:
                return beta

        moves = self.order_moves(board, moves, tt_move, depth)

        self.path_counts[key] = path_seen + 1
        try:
            best_score = -INF
            best_move = None
            orig_alpha = alpha
            for move in moves:
                board.push(move)
                # A forcing check gets searched a ply deeper instead of shorter, since
                # a series of checks needs to be followed to its end (mate, a won
                # material grab, or genuinely nothing) rather than cut off mid-sequence
                # by the normal depth budget — that's exactly the pattern that walked
                # us into a losing repetition instead of a study win a plain search
                # was too shallow to see past.
                if board.is_check() and extensions < MAX_CHECK_EXTENSIONS:
                    child_depth, child_extensions = depth, extensions + 1
                else:
                    child_depth, child_extensions = depth - 1, extensions
                try:
                    score = -self.negamax(board, child_depth, -beta, -alpha, child_extensions)
                finally:
                    board.pop()
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_score > alpha:
                    alpha = best_score
                if alpha >= beta:
                    if not board.is_capture(move) and not move.promotion:
                        self._record_killer(depth, move)
                        self._record_history(board, move, depth)
                    break
        finally:
            if path_seen:
                self.path_counts[key] = path_seen
            else:
                del self.path_counts[key]

        flag = 0
        if best_score <= orig_alpha:
            flag = 2
        elif best_score >= beta:
            flag = 1
        if len(self.tt) > MAX_TT_ENTRIES:
            self.tt.clear()
        self.tt[key] = (depth, best_score, flag, best_move)

        return best_score

    def search(self, board: chess.Board, time_budget: float) -> chess.Move:
        self.deadline = time.monotonic() + time_budget
        self.path_counts.clear()  # defensive: should already be empty after each call
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
                    child_depth = depth if board.is_check() else depth - 1
                    child_extensions = 1 if board.is_check() else 0
                    try:
                        score = -self.negamax(board, child_depth, -beta, -alpha, child_extensions)
                    finally:
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

    # Record that this exact position has now really occurred once more in the
    # game, so the search below can see a real threefold coming even though this
    # freshly-built board has no move history of its own to check against.
    key = chess.polyglot.zobrist_hash(board)
    _engine.history_counts[key] = _engine.history_counts.get(key, 0) + 1

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
