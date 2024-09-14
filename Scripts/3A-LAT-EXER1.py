{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP/T1sPsqqsJzs4rsBfN9uJ",
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
        "<a href=\"https://colab.research.google.com/github/Betinsss/CSST101-3A/blob/main/Scripts/3A-LAT-EXER1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Lab Work: Logic-Based Reasoning Scripts Submission**"
      ],
      "metadata": {
        "id": "KE3l-6iG8fhH"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exercise 1: Propositional Logic in Python**"
      ],
      "metadata": {
        "id": "4bF7ij_98LSw"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3Tr0gfShiScE",
        "outputId": "ade24ccf-fde4-483b-fcae-413ae8b568d1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "It is raining.\n",
            "The match does not light.\n",
            "The match does not light.\n"
          ]
        }
      ],
      "source": [
        "# Represent the statements using propositional logic\n",
        "# p: It rains\n",
        "# q: The ground is wet\n",
        "# r: The match lights\n",
        "\n",
        "# If it rains, the ground will be wet\n",
        "p = True\n",
        "q = p  # Assuming the ground is wet if it rains\n",
        "p_implies_q = not p or q\n",
        "\n",
        "# If the ground is wet, the match will not light\n",
        "q = True  # Assuming the ground is wet\n",
        "r = False  # Assuming the match will not light\n",
        "q_implies_not_r = not q or not r\n",
        "\n",
        "# Evaluate the logical conditions\n",
        "if p:\n",
        "    print(\"It is raining.\")\n",
        "else:\n",
        "    print(\"It is not raining.\")\n",
        "\n",
        "if p_implies_q and q_implies_not_r:\n",
        "    r = False\n",
        "    print(\"The match does not light.\")\n",
        "else:\n",
        "    r = True\n",
        "    print(\"The match lights.\")\n",
        "\n",
        "# Print the final result\n",
        "if r:\n",
        "    print(\"The match lights.\")\n",
        "else:\n",
        "    print(\"The match does not light.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exercise 2: Predicate Logic Representation**"
      ],
      "metadata": {
        "id": "DjaYRNy9q82w"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define predicates\n",
        "def isHuman(person):\n",
        "    # In this scenario, we only know Socrates is a human\n",
        "    return person == \"Socrates\"\n",
        "\n",
        "def isMortal(person):\n",
        "    # All humans are mortal\n",
        "    return isHuman(person)\n",
        "\n",
        "# Define the individual\n",
        "socrates = \"Socrates\"\n",
        "\n",
        "# Check if Socrates is mortal\n",
        "if isMortal(socrates):\n",
        "    print(f\"{socrates} is mortal.\")\n",
        "else:\n",
        "    print(f\"{socrates} is not mortal.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iZUQ0jcgjh54",
        "outputId": "82dd85e9-c69a-4d39-ada7-eec8c7085e08"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Socrates is mortal.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exercise 3: Inference Techniques in Logic-Based Systems**"
      ],
      "metadata": {
        "id": "aXavte2crKvA"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to apply Modus Ponens\n",
        "def apply_modus_ponens(rules, facts):\n",
        "    new_facts = set(facts)  # Set of facts to avoid duplicates\n",
        "\n",
        "    # Repeat inference until no new facts are deduced\n",
        "    while True:\n",
        "        added_fact = False\n",
        "\n",
        "        # Check each rule\n",
        "        for (X, Y) in rules:\n",
        "            if X in new_facts and Y not in new_facts:  # If X is true and Y is not already known\n",
        "                print(f\"Rule applied: If {X} is true, then {Y} is true.\")\n",
        "                new_facts.add(Y)  # Add Y to facts\n",
        "                added_fact = True  # New fact was added\n",
        "\n",
        "        # If no new facts were added, inference is complete\n",
        "        if not added_fact:\n",
        "            break\n",
        "\n",
        "    return new_facts\n",
        "\n",
        "# Example list of rules (in the form of (X, Y) -> \"If X is true, then Y is true\")\n",
        "rules = [\n",
        "    (\"It is raining\", \"The ground is wet\"),\n",
        "    (\"The ground is wet\", \"The match will not light\"),\n",
        "    (\"I study\", \"I pass the exam\"),\n",
        "    (\"I pass the exam\", \"I graduate\")\n",
        "]\n",
        "\n",
        "# Example list of facts\n",
        "facts = [\n",
        "    \"It is raining\",  # Known fact\n",
        "    \"I study\"         # Another known fact\n",
        "]\n",
        "\n",
        "# Apply Modus Ponens to deduce new facts\n",
        "inferred_facts = apply_modus_ponens(rules, facts)\n",
        "\n",
        "# Print the final set of facts\n",
        "print(\"\\nFinal facts deduced:\")\n",
        "for fact in inferred_facts:\n",
        "    print(fact)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SMiZ4cSGlCoH",
        "outputId": "f13f607e-77b4-4ade-b936-e857c4f9cfa5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Rule applied: If It is raining is true, then The ground is wet is true.\n",
            "Rule applied: If The ground is wet is true, then The match will not light is true.\n",
            "Rule applied: If I study is true, then I pass the exam is true.\n",
            "Rule applied: If I pass the exam is true, then I graduate is true.\n",
            "\n",
            "Final facts deduced:\n",
            "The match will not light\n",
            "I pass the exam\n",
            "I study\n",
            "It is raining\n",
            "The ground is wet\n",
            "I graduate\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the rules and facts\n",
        "rules = [\n",
        "    (\"X\", \"Y\")  # Rule: If X is true, then Y is true\n",
        "]\n",
        "\n",
        "facts = [\"X\"]  # Fact: X is true\n",
        "\n",
        "# Function to apply Modus Ponens\n",
        "def apply_modus_ponens(rules, facts):\n",
        "    new_facts = set(facts)\n",
        "    for (P, Q) in rules:\n",
        "        if P in new_facts:\n",
        "            new_facts.add(Q)\n",
        "    return new_facts\n",
        "\n",
        "# Apply Modus Ponens\n",
        "inferred_facts = apply_modus_ponens(rules, facts)\n",
        "\n",
        "# Output the inferred facts\n",
        "print(\"Inferred Facts:\", inferred_facts)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oNEemfmklXB-",
        "outputId": "53fc927e-5ab3-4849-ea74-4c56ea036ccf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Inferred Facts: {'X', 'Y'}\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Exercise 4: Hands-on Lab - Implementing a Logic-Based Model in Python**"
      ],
      "metadata": {
        "id": "Ap1NNYdetmFQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the initial state\n",
        "john_hungry = True  # John is initially hungry\n",
        "john_eats = False   # John has not eaten yet\n",
        "\n",
        "# Define the rules\n",
        "def will_eat(is_hungry):\n",
        "    \"\"\"Rule: If a person is hungry, they will eat.\"\"\"\n",
        "    return is_hungry\n",
        "\n",
        "def no_longer_hungry(eats):\n",
        "    \"\"\"Rule: If a person eats, they will no longer be hungry.\"\"\"\n",
        "    return not eats\n",
        "\n",
        "# Reasoning process\n",
        "while john_hungry:\n",
        "    print(\"John is hungry.\")\n",
        "\n",
        "    # Check if John will eat\n",
        "    if will_eat(john_hungry):\n",
        "        john_eats = True  # John decides to eat\n",
        "        print(\"John decides to eat.\")\n",
        "\n",
        "        # Update hunger status\n",
        "        john_hungry = no_longer_hungry(john_eats)\n",
        "        print(\"John is no longer hungry.\")\n",
        "    else:\n",
        "        print(\"John is still hungry.\")\n",
        "        break  # Exit if John does not eat\n",
        "\n",
        "# Final state\n",
        "if john_hungry:\n",
        "    print(\"John is still hungry.\")\n",
        "else:\n",
        "    print(\"John has eaten and is no longer hungry.\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8yZ-XyRtlny-",
        "outputId": "6201dc13-0158-40f9-881f-ab8335e719b4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "John is hungry.\n",
            "John decides to eat.\n",
            "John is no longer hungry.\n",
            "John has eaten and is no longer hungry.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Assignment: Implement a Logic-Based Model in Python**"
      ],
      "metadata": {
        "id": "eskM-7sZuKsg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the logic-based chatbot for activity recommendations\n",
        "class ActivityRecommender:\n",
        "    def __init__(self):\n",
        "        self.weather_conditions = {\n",
        "            \"sunny\": False,\n",
        "            \"rainy\": False,\n",
        "            \"snowy\": False\n",
        "        }\n",
        "        self.user_preferences = {\n",
        "            \"indoor\": False,\n",
        "            \"outdoor\": False\n",
        "        }\n",
        "\n",
        "    def set_weather(self, weather):\n",
        "        if weather in self.weather_conditions:\n",
        "            self.weather_conditions[weather] = True\n",
        "            print(f\"Weather set to {weather}.\")\n",
        "        else:\n",
        "            print(\"Invalid weather condition.\")\n",
        "\n",
        "    def set_user_preference(self, preference):\n",
        "        if preference in self.user_preferences:\n",
        "            self.user_preferences[preference] = True\n",
        "            print(f\"User preference set to {preference}.\")\n",
        "        else:\n",
        "            print(\"Invalid user preference.\")\n",
        "\n",
        "    def recommend_activity(self):\n",
        "        recommendations = []\n",
        "\n",
        "        # Logic rules for recommendations\n",
        "        if self.weather_conditions[\"sunny\"]:\n",
        "            recommendations.append(\"You can go hiking.\")\n",
        "\n",
        "        if self.weather_conditions[\"rainy\"]:\n",
        "            recommendations.append(\"You can go to the museum.\")\n",
        "\n",
        "        if self.weather_conditions[\"snowy\"]:\n",
        "            recommendations.append(\"You can go skiing.\")\n",
        "\n",
        "        if self.user_preferences[\"indoor\"]:\n",
        "            recommendations.append(\"You can go to the museum.\")\n",
        "\n",
        "        if self.user_preferences[\"outdoor\"]:\n",
        "            if self.weather_conditions[\"sunny\"]:\n",
        "                recommendations.append(\"You can go hiking.\")\n",
        "            if self.weather_conditions[\"snowy\"]:\n",
        "                recommendations.append(\"You can go skiing.\")\n",
        "\n",
        "        # If no recommendations, suggest to stay home\n",
        "        if not recommendations:\n",
        "            recommendations.append(\"No suitable activities found. You might want to stay home.\")\n",
        "\n",
        "        return recommendations\n",
        "\n",
        "# Example usage\n",
        "if __name__ == \"__main__\":\n",
        "    recommender = ActivityRecommender()\n",
        "\n",
        "    # Set weather conditions\n",
        "    recommender.set_weather(\"sunny\")\n",
        "    recommender.set_weather(\"rainy\")  # This will overwrite the previous condition\n",
        "    recommender.set_weather(\"snowy\")   # This will overwrite the previous condition\n",
        "\n",
        "    # Set user preferences\n",
        "    recommender.set_user_preference(\"indoor\")\n",
        "    recommender.set_user_preference(\"outdoor\")  # This will overwrite the previous preference\n",
        "\n",
        "    # Get recommendations\n",
        "    activities = recommender.recommend_activity()\n",
        "    print(\"\\nRecommendations:\")\n",
        "    for activity in activities:\n",
        "        print(activity)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EiTlzYG0m4se",
        "outputId": "9da351ab-5253-4b39-b342-0f497d2307f2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Weather set to sunny.\n",
            "Weather set to rainy.\n",
            "Weather set to snowy.\n",
            "User preference set to indoor.\n",
            "User preference set to outdoor.\n",
            "\n",
            "Recommendations:\n",
            "You can go hiking.\n",
            "You can go to the museum.\n",
            "You can go skiing.\n",
            "You can go to the museum.\n",
            "You can go hiking.\n",
            "You can go skiing.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def get_user_preferences():\n",
        "    \"\"\"Get user preferences for vacation destination.\"\"\"\n",
        "    print(\"Please answer the following questions with 'yes' or 'no':\")\n",
        "    warm_weather = input(\"Do you prefer warm weather? \").strip().lower() == 'yes'\n",
        "    cold_weather = input(\"Do you prefer cold weather? \").strip().lower() == 'yes'\n",
        "    beaches = input(\"Do you prefer beaches? \").strip().lower() == 'yes'\n",
        "    mountains = input(\"Do you prefer mountains? \").strip().lower() == 'yes'\n",
        "\n",
        "    return warm_weather, cold_weather, beaches, mountains\n",
        "\n",
        "def recommend_destination(warm_weather, cold_weather, beaches, mountains):\n",
        "    \"\"\"Recommend a vacation destination based on user preferences.\"\"\"\n",
        "    if warm_weather or beaches:\n",
        "        return \"Hawaii\"\n",
        "    elif cold_weather or mountains:\n",
        "        return \"Switzerland\"\n",
        "    else:\n",
        "        return \"Staycation\"\n",
        "\n",
        "def main():\n",
        "    # Get user preferences\n",
        "    warm_weather, cold_weather, beaches, mountains = get_user_preferences()\n",
        "\n",
        "    # Determine vacation destination\n",
        "    recommendation = recommend_destination(warm_weather, cold_weather, beaches, mountains)\n",
        "\n",
        "    # Provide recommendation\n",
        "    print(f\"Based on your preferences, we recommend: {recommendation}\")\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "y79kcUpbnKc2",
        "outputId": "e4aac8e3-b8cd-46c5-f9bc-2dcb1b3243b5"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Please answer the following questions with 'yes' or 'no':\n",
            "Do you prefer warm weather? no\n",
            "Do you prefer cold weather? yess\n",
            "Do you prefer beaches? yes\n",
            "Do you prefer mountains? yes\n",
            "Based on your preferences, we recommend: Hawaii\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the logic-based chatbot for activity recommendations\n",
        "class ActivityRecommender:\n",
        "    def __init__(self):\n",
        "        self.weather_conditions = {\n",
        "            \"sunny\": False,\n",
        "            \"rainy\": False,\n",
        "            \"snowy\": False\n",
        "        }\n",
        "        self.user_preferences = {\n",
        "            \"indoor\": False,\n",
        "            \"outdoor\": False\n",
        "        }\n",
        "\n",
        "    def set_weather(self, weather):\n",
        "        if weather in self.weather_conditions:\n",
        "            self.weather_conditions[weather] = True\n",
        "            print(f\"Weather set to {weather}.\")\n",
        "        else:\n",
        "            print(\"Invalid weather condition.\")\n",
        "\n",
        "    def set_user_preference(self, preference):\n",
        "        if preference in self.user_preferences:\n",
        "            self.user_preferences[preference] = True\n",
        "            print(f\"User preference set to {preference}.\")\n",
        "        else:\n",
        "            print(\"Invalid user preference.\")\n",
        "\n",
        "    def recommend_activity(self):\n",
        "        recommendations = []\n",
        "\n",
        "        # Logic rules for recommendations\n",
        "        if self.weather_conditions[\"sunny\"]:\n",
        "            recommendations.append(\"You can go hiking.\")\n",
        "\n",
        "        if self.weather_conditions[\"rainy\"]:\n",
        "            recommendations.append(\"You can go to the museum.\")\n",
        "\n",
        "        if self.weather_conditions[\"snowy\"]:\n",
        "            recommendations.append(\"You can go skiing.\")\n",
        "\n",
        "        if self.user_preferences[\"indoor\"]:\n",
        "            recommendations.append(\"You can go to the museum.\")\n",
        "\n",
        "        if self.user_preferences[\"outdoor\"]:\n",
        "            if self.weather_conditions[\"sunny\"]:\n",
        "                recommendations.append(\"You can go hiking.\")\n",
        "            if self.weather_conditions[\"snowy\"]:\n",
        "                recommendations.append(\"You can go skiing.\")\n",
        "\n",
        "        # If no recommendations, suggest to stay home\n",
        "        if not recommendations:\n",
        "            recommendations.append(\"No suitable activities found. You might want to stay home.\")\n",
        "\n",
        "        return recommendations\n",
        "\n",
        "# Example usage\n",
        "if __name__ == \"__main__\":\n",
        "    recommender = ActivityRecommender()\n",
        "\n",
        "    # Set weather conditions\n",
        "    recommender.set_weather(\"sunny\")\n",
        "    recommender.set_user_preference(\"outdoor\")\n",
        "\n",
        "    # Get recommendations\n",
        "    activities = recommender.recommend_activity()\n",
        "    print(\"\\nRecommendations:\")\n",
        "    for activity in activities:\n",
        "        print(activity)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3oGNENdIn2Pd",
        "outputId": "8f32ade0-edfd-4f36-da93-db573951c5bc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Weather set to sunny.\n",
            "User preference set to outdoor.\n",
            "\n",
            "Recommendations:\n",
            "You can go hiking.\n",
            "You can go hiking.\n"
          ]
        }
      ]
    }
  ]
}