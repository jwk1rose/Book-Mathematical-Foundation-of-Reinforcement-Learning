# examples/refactored_assignment_2.py
import sys
import os
from typing import Callable, Dict, Tuple, Optional

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

# project deps (assumed to exist)
from src.grid_world import GridWorld
from examples.arguments import args
from src.utils import create_optimal_path_policy, create_weighted_random_policy, _print_ndarray, _save_or_show, add_state_numbers_to_env

# =========================
# Evaluation core
# =========================

class PolicyEvaluator:
    """Encapsulated evaluator for policy r_pi, P_pi, and value computation."""

    def __init__(self, policy_factory: Callable[[GridWorld], np.ndarray], name: str):
        self.name = name
        temp_world = GridWorld()
        self.policy_matrix = policy_factory(temp_world)
        self.reward_vector, self.transition_matrix = self._derive_policy_markov_dynamics(temp_world)

    def _derive_policy_markov_dynamics(self, world: GridWorld) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute r_pi and P_pi for a fixed policy:
           r_pi[s]     = E[R | s, a ~ pi(·|s)]
           P_pi[s,s']  = Pr(s' | s, a ~ pi(·|s)]
        """
        nS = world.num_states
        r_pi = np.zeros(nS, dtype=float)
        P_pi = np.zeros((nS, nS), dtype=float)

        acts = world.action_space
        s2i = world.state_to_idx
        i2s = world.idx_to_state

        # Accumulate expectation under action distribution
        for s in range(nS):
            xy = i2s[s]
            pmf = self.policy_matrix[s]
            if not np.any(pmf):
                continue
            for a_idx, prob in enumerate(pmf):
                if prob <= 0.0:
                    continue
                nxt_xy, rew = world._get_next_state_and_reward(xy, acts[a_idx])
                s_next = s2i[nxt_xy]
                r_pi[s] += prob * rew
                P_pi[s, s_next] += prob

        return r_pi, P_pi

    def solve_analytically(self, gamma: float) -> np.ndarray:
        """
        Solve (I - gamma P)V = r via linear solve.
        """
        I = np.eye(self.transition_matrix.shape[0], dtype=float)
        return np.linalg.solve(I - gamma * self.transition_matrix, self.reward_vector)

    def solve_iteratively(
        self,
        gamma: float,
        tol: float = 1e-6,
        max_iter: int = 10_000,
    ) -> Tuple[np.ndarray, int]:
        """
        Fixed-point iteration: V_{k+1} = r + gamma P V_k
        """
        v = np.zeros_like(self.reward_vector)
        for k in range(1, max_iter + 1):
            v_next = self.reward_vector + gamma * self.transition_matrix.dot(v)
            if np.max(np.abs(v_next - v)) < tol:
                return v_next, k
            v = v_next
        return v, max_iter

    def run_evaluation(self):
        """
        Export ONLY:
          - policy arrows
          - value grid for closed-form
          - value grid for iterative
        Print r_pi and P_pi to console.
        """
        print("\n" + "#" * 80)
        print(f"Evaluating policy: {self.name}")
        print("#" * 80)

        # print dynamics
        _print_ndarray("r_pi", self.reward_vector)
        _print_ndarray("P_pi", self.transition_matrix)

        # 1) policy arrows
        self._viz_policy(save_path=f"../../../assignment_notes/assignment_2/output/{self.name.lower()}_policy.png")

        gamma = args.discount_rate

        # 2) closed-form value grid
        v_closed = self.solve_analytically(gamma)
        _print_ndarray("V (closed-form)", v_closed)
        self._viz_values(
            v_closed,
            save_path=f"../../../assignment_notes/assignment_2/output/{self.name.lower()}_closed_form_values.png",
        )

        # 3) iterative value grid
        v_iter, steps = self.solve_iteratively(gamma)
        _print_ndarray(f"V (iterative, {steps} iters)", v_iter)
        self._viz_values(
            v_iter,
            save_path=f"../../../assignment_notes/assignment_2/output/{self.name.lower()}_iterative_values.png",
        )

    # ---------- visualization helpers ----------

    def _viz_values(
        self,
        values: np.ndarray,
        precision: int = 2,
        save_path: Optional[str] = None,
    ):
        vis_world = GridWorld()
        vis_world.reset()
        vis_world.render()
        vis_world.add_policy(self.policy_matrix)
        vis_world.add_state_values(values, precision=precision)
        # Do not add state numbers for value plots
        vis_world.render()  # Add this line to refresh the display
        _save_or_show(save_path)

    def _viz_policy(self, save_path: Optional[str] = None):
        vis_world = GridWorld()
        vis_world.reset()
        vis_world.render()
        vis_world.add_policy(self.policy_matrix)
        add_state_numbers_to_env(vis_world)  # Add state numbers only for policy plots
        vis_world.render()  # Add this line to refresh the display
        # no title
        _save_or_show(save_path)


# =========================
# Main
# =========================

def main():
    # ensure output dir
    os.makedirs("../../../assignment_notes/assignment_2/output", exist_ok=True)

    det = PolicyEvaluator(create_optimal_path_policy, "Deterministic")
    det.run_evaluation()

    sto = PolicyEvaluator(create_weighted_random_policy, "Stochastic")
    sto.run_evaluation()


if __name__ == "__main__":
    main()
