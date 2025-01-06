{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Betinsss/CSST101-3A/blob/main/Scripts/LAT-MDP.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Google Colab Code for Markov Decision Process**"
      ],
      "metadata": {
        "id": "m5BTw1zFER5-"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Step 1: Set Up the Environment**"
      ],
      "metadata": {
        "id": "GMdOWBBzEVhL"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3IoZhyDh6GXO"
      },
      "outputs": [],
      "source": [
        "# Install required libraries\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# Define the grid environment parameters\n",
        "grid_size = 5\n",
        "start_state = (0, 0)\n",
        "goal_state = (4, 4)\n",
        "penalty_states = [(2, 2), (3, 3)]\n",
        "\n",
        "# Rewards for different states\n",
        "reward_goal = 10\n",
        "reward_penalty = -10\n",
        "reward_step = -1\n",
        "\n",
        "# Define actions and their effects\n",
        "actions = [\"U\", \"D\", \"L\", \"R\"]\n",
        "action_effects = {\n",
        "    \"U\": (-1, 0),  # Up\n",
        "    \"D\": (1, 0),   # Down\n",
        "    \"L\": (0, -1),  # Left\n",
        "    \"R\": (0, 1)    # Right\n",
        "}\n",
        "\n",
        "# Transition probabilities\n",
        "transition_probabilities = {\n",
        "    \"intended\": 0.8,\n",
        "    \"left\": 0.1,\n",
        "    \"right\": 0.1\n",
        "}\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Step 2: Define Helper Functions**"
      ],
      "metadata": {
        "id": "mJ_f-4DYEid-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def is_valid(state):\n",
        "    \"\"\"Check if the state is within grid boundaries.\"\"\"\n",
        "    x, y = state\n",
        "    return 0 <= x < grid_size and 0 <= y < grid_size\n",
        "\n",
        "def get_next_state(state, action):\n",
        "    \"\"\"Get the next state after taking an action.\"\"\"\n",
        "    dx, dy = action_effects[action]\n",
        "    next_state = (state[0] + dx, state[1] + dy)\n",
        "    return next_state if is_valid(next_state) else state\n",
        "\n",
        "def get_reward(state):\n",
        "    \"\"\"Return the reward for a given state.\"\"\"\n",
        "    if state == goal_state:\n",
        "        return reward_goal\n",
        "    if state in penalty_states:\n",
        "        return reward_penalty\n",
        "    return reward_step\n"
      ],
      "metadata": {
        "id": "es0JEER1ElRI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Step 3: Value Iteration Implementation**"
      ],
      "metadata": {
        "id": "YxPVxWomEpU1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def value_iteration(discount_factor=0.9, theta=1e-4):\n",
        "    \"\"\"Perform Value Iteration to find the optimal policy.\"\"\"\n",
        "    # Initialize value function and policy\n",
        "    V = np.zeros((grid_size, grid_size))\n",
        "    policy = np.full((grid_size, grid_size), '', dtype=object)\n",
        "\n",
        "    while True:\n",
        "        delta = 0  # Maximum change in value function\n",
        "        new_V = np.copy(V)\n",
        "\n",
        "        for x in range(grid_size):\n",
        "            for y in range(grid_size):\n",
        "                state = (x, y)\n",
        "                if state == goal_state:\n",
        "                    continue  # Skip goal state\n",
        "\n",
        "                action_values = []\n",
        "                for action in actions:\n",
        "                    value = 0\n",
        "                    for prob_type, prob in transition_probabilities.items():\n",
        "                        if prob_type == \"intended\":\n",
        "                            next_action = action\n",
        "                        elif prob_type == \"left\":\n",
        "                            next_action = actions[(actions.index(action) - 1) % 4]\n",
        "                        elif prob_type == \"right\":\n",
        "                            next_action = actions[(actions.index(action) + 1) % 4]\n",
        "\n",
        "                        next_state = get_next_state(state, next_action)\n",
        "                        value += prob * (get_reward(next_state) + discount_factor * V[next_state])\n",
        "                    action_values.append(value)\n",
        "\n",
        "                # Update value function and policy\n",
        "                best_action_value = max(action_values)\n",
        "                new_V[state] = best_action_value\n",
        "                policy[state] = actions[np.argmax(action_values)]\n",
        "                delta = max(delta, abs(new_V[state] - V[state]))\n",
        "\n",
        "        V = new_V\n",
        "        if delta < theta:  # Convergence condition\n",
        "            break\n",
        "\n",
        "    return V, policy\n"
      ],
      "metadata": {
        "id": "zIY6eJrJEqh8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Step 4: Run Value Iteration**"
      ],
      "metadata": {
        "id": "NDGGV3VrEty8"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Run the Value Iteration algorithm\n",
        "optimal_values, optimal_policy = value_iteration()\n",
        "\n",
        "# Display the results\n",
        "print(\"Optimal Value Function:\")\n",
        "print(optimal_values)\n",
        "\n",
        "print(\"\\nOptimal Policy:\")\n",
        "for row in optimal_policy:\n",
        "    print(row)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "r4aHN5HhEv-z",
        "outputId": "3c5ce72e-da77-4458-a5f7-be3bcaa0cdc9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Optimal Value Function:\n",
            "[[-3.69041572 -2.81400612 -1.7063552  -0.41597141  1.0764551 ]\n",
            " [-2.81400612 -1.98954293 -0.67081883  0.91919839  2.80141652]\n",
            " [-1.7063552  -0.67081883  0.19087519  1.82171427  5.03029156]\n",
            " [-0.41597141  0.91919839  1.82171427  4.94211829  7.79751667]\n",
            " [ 1.0764551   2.80141652  5.03029156  7.79751667  0.        ]]\n",
            "\n",
            "Optimal Policy:\n",
            "['D' 'R' 'R' 'R' 'D']\n",
            "['D' 'D' 'R' 'R' 'D']\n",
            "['D' 'D' 'D' 'R' 'D']\n",
            "['D' 'D' 'D' 'D' 'D']\n",
            "['R' 'R' 'R' 'R' '']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Step 5: Visualization**"
      ],
      "metadata": {
        "id": "NQKvidPvExuN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def plot_policy(policy):\n",
        "    \"\"\"Visualize the optimal policy.\"\"\"\n",
        "    fig, ax = plt.subplots(figsize=(6, 6))\n",
        "    for x in range(grid_size):\n",
        "        for y in range(grid_size):\n",
        "            if (x, y) == goal_state:\n",
        "                ax.text(y, x, \"G\", ha='center', va='center', color='green', fontsize=14)\n",
        "            elif (x, y) in penalty_states:\n",
        "                ax.text(y, x, \"X\", ha='center', va='center', color='red', fontsize=14)\n",
        "            else:\n",
        "                ax.text(y, x, policy[(x, y)], ha='center', va='center', fontsize=12)\n",
        "\n",
        "    ax.set_xticks(np.arange(-0.5, grid_size, 1))\n",
        "    ax.set_yticks(np.arange(-0.5, grid_size, 1))\n",
        "    ax.grid(True)\n",
        "    ax.set_xlim(-0.5, grid_size - 0.5)\n",
        "    ax.set_ylim(-0.5, grid_size - 0.5)\n",
        "    plt.gca().invert_yaxis()\n",
        "    plt.title(\"Optimal Policy Visualization\")\n",
        "    plt.show()\n",
        "\n",
        "plot_policy(optimal_policy)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 545
        },
        "id": "sGpooYLME0R8",
        "outputId": "98fa79f7-85e9-4115-f078-57a3507bb3ae"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 600x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAhsAAAIQCAYAAAAo4Kx+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABB7UlEQVR4nO3de5zOdf7/8edljBmDGcxgTpqZHFMmIaUtjSKZFaITIkLZKL5ZS23JYdupTcwWSbWMnGq3jdQS4zCsHEJZp9hYRg6DcbicmnGZ+fz+8HPVmINR8/58xlyP++123dbneL2u1/XZT8/5fN7Xdbksy7IEAABgSDmnCwAAAGUbYQMAABhF2AAAAEYRNgAAgFGEDQAAYBRhAwAAGEXYAAAARhE2AACAUYQNAABgFGEDKIaUlBS5XC7t3bvXZ547NjZWvXv39k6npaXJ5XIpLS3N1jqKy8n36Ep1JCQkKCEhwfZanHpe4HKEDVyTtm3bpscff1xRUVEKCAhQZGSkevTooW3btv2q/f75z3/WvHnzSqZIm40aNUoul8v7CAoKUqNGjfTSSy/p1KlTTpd31Tp27KigoCCdPn260HV69OihChUq6NixYzZWVrps375do0aNcjxkAUUhbOCa8+mnn6pp06ZaunSp+vTpo3feeUd9+/bV8uXL1bRpU82dO/cX77uwsNGzZ0/9+OOPiomJ+RWV22Py5MmaMWOGxo8fr4YNG+rVV1/V/fffr1/7M0itWrXSjz/+qFatWpVQpUXr0aOHfvzxx0Lfz3Pnzumzzz7T/fffr9DQ0FL9Hi1evFiLFy82su/t27dr9OjRBYYNk88LXI3yThcAXI3du3erZ8+euv7667Vy5UrVqFHDu2zw4MG666671LNnT23evFnXX399iT2vn5+f/Pz8Smx/Jj300EMKCwuTJA0YMEBdu3bVp59+qrVr16ply5a/eL/lypVTYGBgSZV5RR07dlSVKlU0e/Zs9erVK9/yzz77TGfPnlWPHj0kle73qEKFCj71vMDluLKBa8obb7yhc+fO6b333ssTNCQpLCxMU6ZM0dmzZ/WXv/zFO//S7YUdO3bokUceUXBwsEJDQzV48GBlZWV513O5XDp79qymT5/uvRVxacxCQffhY2Nj1aFDB6Wlpal58+aqWLGiGjdu7B3T8Omnn6px48YKDAxUs2bN9O233+apd/Pmzerdu7euv/56BQYGKjw8XE8++WSJ3xK45557JEl79uyRJJ09e1ZDhw5V7dq1FRAQoAYNGmjcuHFXvPJR2JiNdevWKTExUdWqVVOlSpUUHx+vv/71r5KkadOmyeVy5Xvt0sWrSH5+fjpw4ECBz1exYkV16dJFS5cu1ZEjR/Itnz17tqpUqaKOHTtKKvg92rBhg9q1a6ewsDBVrFhRcXFxevLJJ6/4mvbu3SuXy6WUlBTvvF/zfl0+diI2NjbPLa+fPy7Vkp6ermeeeUYNGjRQxYoVFRoaqocffjjP60tJSdHDDz8sSWrdunW+fRQ0ZuPIkSPq27evatWqpcDAQN18882aPn16ga9/3Lhxeu+991SnTh0FBATo1ltv1fr166/4eoHLcWUD15TPP/9csbGxuuuuuwpc3qpVK8XGxupf//pXvmWPPPKIYmNjlZSUpLVr1+qtt97SiRMn9OGHH0qSZsyYoX79+qlFixZ66qmnJEl16tQpsp5du3ape/fuevrpp/X4449r3LhxeuCBB/Tuu+/qxRdf1DPPPCNJSkpK0iOPPKKdO3eqXLmLGT81NVX/+9//1KdPH4WHh2vbtm167733tG3bNq1du1Yul+sX9+nndu/eLUkKDQ2VZVnq2LGjli9frr59+6pJkyZatGiRhg0bpgMHDmjChAlXte/U1FR16NBBERERGjx4sMLDw/Xdd9/piy++0ODBg/XQQw9p4MCBmjVrlm655ZY8286aNUsJCQmKiooqdP89evTQ9OnT9fe//12DBg3yzj9+/LgWLVqkbt26qWLFigVue+TIEd13332qUaOGRowYoapVq2rv3r369NNPr+o1/vy1ltT7lZycrDNnzuSZN2HCBG3atEmhoaGSpPXr12v16tV67LHHFB0drb1792ry5MlKSEjQ9u3bFRQUpFatWum5557TW2+9pRdffFE33HCDJHn/93I//vijEhIStGvXLg0aNEhxcXH6xz/+od69e+vkyZMaPHhwnvVnz56t06dP6+mnn5bL5dJf/vIXdenSRf/73//k7+9/Ne2Dr7OAa8TJkyctSVanTp2KXK9jx46WJOvUqVOWZVnWK6+8YkmyOnbsmGe9Z555xpJk/ec///HOq1SpkvXEE0/k2+e0adMsSdaePXu882JiYixJ1urVq73zFi1aZEmyKlasaKWnp3vnT5kyxZJkLV++3Dvv3Llz+Z5nzpw5liRr5cqVRT53QS69zp07d1pHjx619uzZY02ZMsUKCAiwatWqZZ09e9aaN2+eJcn605/+lGfbhx56yHK5XNauXbvyvL6f92L58uV5XsOFCxesuLg4KyYmxjpx4kSe/eXm5nr/3a1bNysyMtLKycnxzvvmm28sSda0adOKfE0XLlywIiIirJYtW+aZ/+6771qSrEWLFnnnXd6nuXPnWpKs9evXF7r/y1/TJXv27MlX3695v+6++27r7rvvLrSOv//975Yka8yYMUU+35o1ayxJ1ocffuid949//KPA11DQ8yYnJ1uSrJkzZ3rnnT9/3mrZsqVVuXJl7/9nLr3+0NBQ6/jx4951P/vsM0uS9fnnnxf6WoCCcBsF14xLn0qoUqVKketdWn75JzAGDhyYZ/rZZ5+VJC1YsOAX19SoUaM84yBuu+02SRdvXVx33XX55v/vf//zzvv5X+RZWVnKzMzU7bffLkn65ptvfnFNDRo0UI0aNRQXF6enn35adevW1b/+9S8FBQVpwYIF8vPz03PPPZdnm6FDh8qyLC1cuLDYz/Ptt99qz549GjJkiKpWrZpn2c//yu/Vq5cOHjyo5cuXe+fNmjVLFStWVNeuXYt8Dj8/Pz322GNas2ZNntsHs2fPVq1atXTvvfcWuu2lmr744gt5PJ5iv67CmHq/tm/frieffFKdOnXSSy+9VODzeTweHTt2THXr1lXVqlV/8fMtWLBA4eHh6tatm3eev7+/nnvuOZ05c0YrVqzIs/6jjz6qatWqeacvXVH8+XEMFAdhA9eMSyGiqI9C/nz55aGkXr16eabr1KmjcuXK/aqPDP48UEhSSEiIJKl27doFzj9x4oR33vHjxzV48GDVqlVLFStW9AYESXK73b+4pn/+859KTU1VWlqadu3apa1bt6pZs2aSLo4DiIyMzNebS5fd09PTi/08l27P3HTTTUWu17ZtW0VERGjWrFmSpNzcXM2ZM0edOnW6YnCU5B0AOnv2bEnS/v379e9//1uPPfZYkQNC7777bnXt2lWjR49WWFiYOnXqpGnTpik7O7tYr+9yJt6vU6dOqUuXLoqKitKHH36YJ6T9+OOPGjlypHdsTVhYmGrUqKGTJ0/+4udLT09XvXr1vLfyLins/b/8+L4UPH5+HAPFwZgNXDNCQkIUERGhzZs3F7ne5s2bFRUVpeDg4CLXK4kxEYX9x66w+dbPBmE+8sgjWr16tYYNG6YmTZqocuXKys3N1f3336/c3NxfXFOrVq28n0YpDfz8/NS9e3e9//77euedd/TVV1/p4MGDevzxx4u1fbNmzdSwYUPNmTNHL774oubMmSPLsrwhpDAul0uffPKJ1q5dq88//1yLFi3Sk08+qTfffFNr165V5cqVCz0GcnJy8s0z8X717t1bBw8e1Ndff53veH322Wc1bdo0DRkyRC1btlRISIhcLpcee+yxX3V8XI3iHMdAcXBlA9eUDh06aM+ePVq1alWBy//9739r79696tChQ75l33//fZ7pXbt2KTc3V7Gxsd55JTUo80pOnDihpUuXasSIERo9erQefPBBtW3btkQ/rluQmJgYHTx4MN/VoR07dniXF9elwbNbt2694rq9evXSqVOn9Pnnn2vWrFmqUaOG2rVrV+zn6tGjh7Zu3arNmzdr9uzZqlevnm699dZibXv77bfr1Vdf1YYNGzRr1ixt27ZNH330kaSf/lI/efJknm0u/wvfxPv12muvad68efrwww/VsGHDfMs/+eQTPfHEE3rzzTf10EMPqW3btrrzzjvz1Xo1x2xMTIy+//77fGHll7z/wNUgbOCaMmzYMFWsWFFPP/10vo8cHj9+XAMGDFBQUJCGDRuWb9tJkyblmX777bclSe3bt/fOq1SpUr6TuQmX/mK8/C/E5ORko8+bmJionJwcTZw4Mc/8CRMmyOVy5enFlTRt2lRxcXFKTk7O17PLX1d8fLzi4+P1wQcf6J///Kcee+wxlS9f/Aurl65ijBw5Ups2bbriVQ3pYkC4vI4mTZpIkvdWSkxMjPz8/LRy5co8673zzjt5pkv6/VqyZIleeukl/fGPf1Tnzp0LXMfPzy/f87399tv5rrpUqlRJUv7AVJDExERlZGTo448/9s67cOGC3n77bVWuXFl333331b0QoJi4jYJrSr169TR9+nT16NFDjRs3Vt++fRUXF6e9e/fqb3/7mzIzMzVnzpwCP7K6Z88edezYUffff7/WrFmjmTNnqnv37rr55pu96zRr1kxLlizR+PHjFRkZqbi4OO/gzpIUHBysVq1a6S9/+Ys8Ho+ioqK0ePFi73dhmPLAAw+odevW+uMf/6i9e/fq5ptv1uLFi/XZZ59pyJAhV/yo78+VK1dOkydP1gMPPKAmTZqoT58+ioiI0I4dO7Rt2zYtWrQoz/q9evXS73//e0kq9i2US+Li4nTHHXfos88+k6RihY3p06frnXfe0YMPPqg6dero9OnTev/99xUcHKzExERJF2/NPfzww3r77bflcrlUp04dffHFF/m+16Ok369u3bqpRo0aqlevnmbOnJlnWdu2bVWrVi116NBBM2bMUEhIiBo1aqQ1a9ZoyZIl3o/GXtKkSRP5+fnp9ddfl9vtVkBAgO655x7VrFkz3/M+9dRTmjJlinr37q2NGzcqNjZWn3zyib766islJycXawwN8Is49jkY4FfYvHmz1a1bNysiIsLy9/e3wsPDrW7dullbtmzJt+6lj4Ru377deuihh6wqVapY1apVswYNGmT9+OOPedbdsWOH1apVK6tixYqWJO9HPwv76Otvf/vbfM8nyRo4cGCeeZc+SvjGG2945+3fv9968MEHrapVq1ohISHWww8/bB08eNCSZL3yyive9a72o69Hjx4tcr3Tp09b//d//2dFRkZa/v7+Vr169aw33ngjz8dVL72+oj76esmqVaustm3bWlWqVLEqVapkxcfHW2+//Xa+5z106JDl5+dn1a9fv8j6CjNp0iRLktWiRYsCl1/ep2+++cbq1q2bdd1111kBAQFWzZo1rQ4dOlgbNmzIs93Ro0etrl27WkFBQVa1atWsp59+2tq6dWu+j77+mvfr8o+gSir0cam/J06csPr06WOFhYVZlStXttq1a2ft2LEj3/tiWZb1/vvvW9dff73l5+eXZx8FfeT28OHD3v1WqFDBaty4cb6PIBd0vP689p+/XqA4XJbFSB+UbaNGjdLo0aN19OjRUjVw0tdkZmYqIiJCI0eO1Msvv+x0OQBsxJgNALZISUlRTk6Oevbs6XQpAGzGmA0ARi1btkzbt2/Xq6++qs6dO+f59A8A30DYAGDUmDFjtHr1av3mN7/xfgIIgG8xdhvl+PHj6tGjh4KDg1W1alX17ds33w8PXS4hISHfLyAOGDDAVInwEaNGjZJlWYzXcEhaWprOnz+v5cuXF/mjawDKLmMDRNu3b69Dhw5pypQp8ng86tOnj2699VbvVw4XJCEhQfXr19eYMWO884KCgq74TZAAAKD0MnIb5bvvvtOXX36p9evXq3nz5pIufhlNYmKixo0bp8jIyEK3DQoKUnh4uImyAACAA4yEjTVr1qhq1areoCFJbdq0Ubly5bRu3To9+OCDhW47a9YszZw5U+Hh4XrggQf08ssvKygoqND1s7Oz8/ywUm5uro4fP67Q0FDbvnoaAICywLIsnT59WpGRkfl+sO/XMBI2MjIy8n17Xfny5VW9enVlZGQUul337t0VExOjyMhIbd68WcOHD9fOnTv16aefFrpNUlKSRo8eXWK1AwDg63744QdFR0eX2P6uKmyMGDFCr7/+epHrfPfdd7+4mKeeesr778aNGysiIkL33nuvdu/eXejXKL/wwgt6/vnnvdNut1vXXXed/vvf/6p69eq/uBYUn8fj0fLly9W6dWv5+/s7XY5PoOf2o+f2o+f2O378uOrXr1/iX11/VWFj6NCh6t27d5HrXH/99QoPD8/32wIXLlzQ8ePHr2o8xqXfpNi1a1ehYSMgIEABAQH55levXj3fbwjADI/Ho6CgIIWGhnJCsAk9tx89tx89d05JD0O4qrBRo0YN1ahR44rrtWzZUidPntTGjRvVrFkzSRe/2Cc3N/eqftRq06ZNkqSIiIirKRMAAJQiRr5n44YbbtD999+v/v376+uvv9ZXX32lQYMG6bHHHvN+EuXAgQNq2LChvv76a0nS7t27NXbsWG3cuFF79+7V/Pnz1atXL7Vq1Urx8fEmygQAADYw9qVes2bNUsOGDXXvvfcqMTFRd955p9577z3vco/Ho507d+rcuXOSpAoVKmjJkiW677771LBhQw0dOlRdu3bV559/bqpEAABgA2NfV169evUiv8ArNjZWP/8+sdq1a2vFihWmygEAAA7hV18BAIBRhA0AAGAUYQMAABhF2AAAAEYRNgAAgFGEDQAAYBRhAwAAGEXYAAAARhE2AACAUYQNAABgFGEDAAAYRdgAAABGETYAAIBRhA0AAGAUYQMAABhF2AAAAEYRNgAAgFGEDQAAYBRhAwAAGEXYAAAARhE2AACAUYQNAABgFGEDAAAYRdgAAABGETYAAIBRhA0AAGAUYQMAABhF2AAAAEYRNgAAgFGEDQAAYBRhAwAAGEXYAAAARhE2HJSSkiKXy+V9BAYGKjIyUu3atdNbb72l06dPO11imXR538uXL6+oqCj17t1bBw4ccLq8Mome24+e24vzedHKO10ApDFjxiguLk4ej0cZGRlKS0vTkCFDNH78eM2fP1/x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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Expected Outputs**\n",
        "1. **Optimal Value Function:** Displays a 5x5 grid with the maximum cumulative reward for each state.\n",
        "\n",
        "2. **Optimal Policy:** Prints the best action (↑, ↓, ←, →) for each state in a 5x5 grid.\n",
        "\n",
        "3. **Visualization:** A plot of the grid, showing:\n",
        "\n",
        "  * G for the goal state.\n",
        "  * X for penalty states.\n",
        "  * Arrows for the best actions.\n",
        "\n"
      ],
      "metadata": {
        "id": "XR4gTFj9E3DM"
      }
    }
  ]
}