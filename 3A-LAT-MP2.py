{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOoQtm3dAa5w7JjsmZfC/Ma",
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
        "<a href=\"https://colab.research.google.com/github/Betinsss/CSST101-CS3A/blob/main/3A_LAT_MP2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "1. Propositional Logic Operations:"
      ],
      "metadata": {
        "id": "Pv03aGUZJNKg"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rWtqWCz41XdR"
      },
      "outputs": [],
      "source": [
        "def AND(p, q):\n",
        "    return p and q\n",
        "\n",
        "def OR(p, q):\n",
        "    return p or q\n",
        "\n",
        "def NOT(p):\n",
        "    return not p\n",
        "\n",
        "def IMPLIES(p, q):\n",
        "    return (not p) or q"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "2. Testing Propositional Logic Functions"
      ],
      "metadata": {
        "id": "TEjcLVpfJPOY"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(AND(True, False))  # Expected: False\n",
        "print(OR(True, False))   # Expected: True\n",
        "print(NOT(True))          # Expected: False\n",
        "print(IMPLIES(True, False))  # Expected: False"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IaVszl6K1gei",
        "outputId": "f00e8e1a-0aa8-4892-bd1c-197bc43c17a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "False\n",
            "True\n",
            "False\n",
            "False\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "3. Evaluate Logical Statements"
      ],
      "metadata": {
        "id": "xBKHfWoWJT0v"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def evaluate(statement, values):\n",
        "    statement = statement.replace('and', ' and ').replace('or', ' or ').replace('not', ' not ').replace('=>', ' or not ')\n",
        "    for var, val in values.items():\n",
        "        statement = statement.replace(var, str(val))\n",
        "    return eval(statement)"
      ],
      "metadata": {
        "id": "CNgioV3c1oWK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "4. Testing the Evaluation Function"
      ],
      "metadata": {
        "id": "_r0naQ7tJbdw"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(evaluate('A and B', {'A': True, 'B': False}))  # Expected: False\n",
        "print(evaluate('A or B', {'A': True, 'B': False}))   # Expected: True\n",
        "print(evaluate('not A', {'A': True}))                 # Expected: False\n",
        "print(evaluate('A => B', {'A': True, 'B': False}))    # Expected: False\n"
      ],
      "metadata": {
        "id": "6KrWQ9Yk1roK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d2c35298-fad6-4de8-d4b9-2045c7da2c9f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "False\n",
            "True\n",
            "False\n",
            "True\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "5. Extend to Predicate Logic"
      ],
      "metadata": {
        "id": "cAoUYFz5PsH_"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def forall(predicate, domain):\n",
        "    \"\"\"Evaluate a predicate for all elements in the domain.\"\"\"\n",
        "    return all(predicate(x) for x in domain)\n",
        "\n",
        "def exists(predicate, domain):\n",
        "    \"\"\"Evaluate if there exists an element in the domain for which the predicate is true.\"\"\"\n",
        "    return any(predicate(x) for x in domain)"
      ],
      "metadata": {
        "id": "xwj46axQ2BeL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "6. Testing Predicate Logic Functions"
      ],
      "metadata": {
        "id": "HMR11ViuPt2e"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "predicate = lambda x: x > 0\n",
        "domain = [1, 2, 3, -1, -2]\n",
        "\n",
        "print(forall(predicate, domain))  # Expected: False (because not all positive)\n",
        "print(exists(predicate, domain))  # Expected: True (because is there are positive numbers)"
      ],
      "metadata": {
        "id": "aEHrXvFs2COi",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "91a7973d-2d46-4585-872a-d24046d9511d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "False\n",
            "True\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "7. AI Agent Development"
      ],
      "metadata": {
        "id": "CQpFmoQgPwTm"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class SimpleAIAgent:\n",
        "    def __init__(self):\n",
        "        self.condition = True  # Example condition\n",
        "\n",
        "    def make_decision(self):\n",
        "        if AND(self.condition, True):  # Simple logic\n",
        "            return 'Take action A'\n",
        "        else:\n",
        "            return 'Take action B'\n",
        "\n",
        "# Create an AI agent instance\n",
        "agent = SimpleAIAgent()\n",
        "\n",
        "# Make a decision\n",
        "print(agent.make_decision())  # Expected: 'Take action A'"
      ],
      "metadata": {
        "id": "w3j78PY62H3C",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d9df3317-312c-46ae-bb2c-2e13d5a41e35"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Take action A\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# AI Agent Logic\n",
        "\n",
        "def ai_agent(weather, temperature):\n",
        "    \"\"\"\n",
        "    AI agent that decides whether to go outside based on weather and temperature.\n",
        "\n",
        "    Args:\n",
        "    weather (str): The weather condition ('sunny', 'rainy', 'cloudy').\n",
        "    temperature (int): The current temperature in degrees Celsius.\n",
        "\n",
        "    Returns:\n",
        "    str: The decision made by the AI agent ('Go outside', 'Stay inside', or 'Undecided').\n",
        "    \"\"\"\n",
        "    # Define logical conditions for decision-making\n",
        "    sunny = (weather == 'sunny')\n",
        "    rainy = (weather == 'rainy')\n",
        "    cloudy = (weather == 'cloudy')\n",
        "    cold = (temperature < 10)   # Consider cold if temperature is below 10°C\n",
        "    hot = (temperature > 30)    # Consider hot if temperature is above 30°C\n",
        "    moderate = (10 <= temperature <= 30)\n",
        "\n",
        "    # Decision logic using propositional logic operations\n",
        "    # Rule 1: If it's sunny and not cold, the agent will go outside, unless it's too hot.\n",
        "    if IMPLIES(sunny, AND(NOT(cold), NOT(hot))):\n",
        "        return \"Go outside\"\n",
        "\n",
        "    # Rule 2: If it's rainy or too cold, the agent will stay inside.\n",
        "    if OR(rainy, cold):\n",
        "        return \"Stay inside\"\n",
        "\n",
        "    # Rule 3: In moderate temperature and cloudy weather, the agent is undecided.\n",
        "    if AND(cloudy, moderate):\n",
        "        return \"Undecided\"\n",
        "\n",
        "    # Default rule: Stay inside in other cases\n",
        "    return \"Stay inside\"\n",
        "\n",
        "\n",
        "# Test cases\n",
        "\n",
        "def run_ai_agent_tests():\n",
        "    \"\"\"\n",
        "    Test the AI agent decision-making in various scenarios.\n",
        "    \"\"\"\n",
        "    test_cases = [\n",
        "        ('sunny', 25),    # Moderate sunny day: Go outside\n",
        "        ('sunny', 35),    # Hot sunny day: Stay inside\n",
        "        ('rainy', 5),     # Cold rainy day: Stay inside\n",
        "        ('cloudy', 15),   # Moderate cloudy day: Undecided\n",
        "        ('cloudy', 5),    # Cold cloudy day: Stay inside\n",
        "        ('sunny', 5),     # Cold sunny day: Stay inside\n",
        "        ('rainy', 15),    # Rainy moderate day: Stay inside\n",
        "    ]\n",
        "\n",
        "    for weather, temp in test_cases:\n",
        "        result = ai_agent(weather, temp)\n",
        "        print(f\"Weather: {weather}, Temperature: {temp}°C -> Decision: {result}\")\n",
        "\n",
        "# Running test cases\n",
        "run_ai_agent_tests()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cBNUjqTaRoP2",
        "outputId": "1e47a5be-28eb-4528-d8e8-64ac2a4c9a9c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Weather: sunny, Temperature: 25°C -> Decision: Go outside\n",
            "Weather: sunny, Temperature: 35°C -> Decision: Stay inside\n",
            "Weather: rainy, Temperature: 5°C -> Decision: Go outside\n",
            "Weather: cloudy, Temperature: 15°C -> Decision: Go outside\n",
            "Weather: cloudy, Temperature: 5°C -> Decision: Go outside\n",
            "Weather: sunny, Temperature: 5°C -> Decision: Stay inside\n",
            "Weather: rainy, Temperature: 15°C -> Decision: Go outside\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Explanation of the Decision Logic:\n",
        "\n",
        "Rule 1: The agent will decide to \"Go outside\" if the weather is sunny and the temperature is not too cold or hot.\n",
        "\n",
        "Rule 2: If the weather is rainy or too cold, the agent will decide to \"Stay inside.\"\n",
        "\n",
        "Rule 3: If the temperature is moderate and the weather is cloudy, the agent will remain \"Undecided.\"\n",
        "\n",
        "Default Rule: In any other situation, the agent will \"Stay inside.\"\n",
        "\n"
      ],
      "metadata": {
        "id": "nwdmXCE1SVh2"
      }
    }
  ]
}
