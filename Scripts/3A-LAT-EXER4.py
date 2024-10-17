{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNHGTyenv0dP/QoUt8nQC9r",
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
        "<a href=\"https://colab.research.google.com/github/Betinsss/CSST101-3A/blob/main/Scripts/3A-LAT-EXER4.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#**Implementation of Bayesian Networks in Google Colab**"
      ],
      "metadata": {
        "id": "w5R64Bnf2jJM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Exercise 1: Setting Up the Environment**\n",
        "\n"
      ],
      "metadata": {
        "id": "rTtrEOzw2pUM"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1. Install the Required Library:**"
      ],
      "metadata": {
        "id": "cGtpCwTv2xzT"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a3JQA1JN2TXh",
        "outputId": "750710ca-f6ff-40e7-e4cd-9c6fc9c10da4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pgmpy in /usr/local/lib/python3.10/dist-packages (0.1.26)\n",
            "Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from pgmpy) (3.3)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from pgmpy) (1.26.4)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from pgmpy) (1.13.1)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (from pgmpy) (1.5.2)\n",
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from pgmpy) (2.2.2)\n",
            "Requirement already satisfied: pyparsing in /usr/local/lib/python3.10/dist-packages (from pgmpy) (3.1.4)\n",
            "Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (from pgmpy) (2.4.1+cu121)\n",
            "Requirement already satisfied: statsmodels in /usr/local/lib/python3.10/dist-packages (from pgmpy) (0.14.4)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from pgmpy) (4.66.5)\n",
            "Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from pgmpy) (1.4.2)\n",
            "Requirement already satisfied: opt-einsum in /usr/local/lib/python3.10/dist-packages (from pgmpy) (3.4.0)\n",
            "Requirement already satisfied: xgboost in /usr/local/lib/python3.10/dist-packages (from pgmpy) (2.1.1)\n",
            "Requirement already satisfied: google-generativeai in /usr/local/lib/python3.10/dist-packages (from pgmpy) (0.7.2)\n",
            "Requirement already satisfied: google-ai-generativelanguage==0.6.6 in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (0.6.6)\n",
            "Requirement already satisfied: google-api-core in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (2.19.2)\n",
            "Requirement already satisfied: google-api-python-client in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (2.137.0)\n",
            "Requirement already satisfied: google-auth>=2.15.0 in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (2.27.0)\n",
            "Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (3.20.3)\n",
            "Requirement already satisfied: pydantic in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (2.9.2)\n",
            "Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from google-generativeai->pgmpy) (4.12.2)\n",
            "Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in /usr/local/lib/python3.10/dist-packages (from google-ai-generativelanguage==0.6.6->google-generativeai->pgmpy) (1.24.0)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->pgmpy) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->pgmpy) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->pgmpy) (2024.2)\n",
            "Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn->pgmpy) (3.5.0)\n",
            "Requirement already satisfied: patsy>=0.5.6 in /usr/local/lib/python3.10/dist-packages (from statsmodels->pgmpy) (0.5.6)\n",
            "Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.10/dist-packages (from statsmodels->pgmpy) (24.1)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch->pgmpy) (3.16.1)\n",
            "Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch->pgmpy) (1.13.3)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch->pgmpy) (3.1.4)\n",
            "Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch->pgmpy) (2024.6.1)\n",
            "Requirement already satisfied: nvidia-nccl-cu12 in /usr/local/lib/python3.10/dist-packages (from xgboost->pgmpy) (2.23.4)\n",
            "Requirement already satisfied: googleapis-common-protos<2.0.dev0,>=1.56.2 in /usr/local/lib/python3.10/dist-packages (from google-api-core->google-generativeai->pgmpy) (1.65.0)\n",
            "Requirement already satisfied: requests<3.0.0.dev0,>=2.18.0 in /usr/local/lib/python3.10/dist-packages (from google-api-core->google-generativeai->pgmpy) (2.32.3)\n",
            "Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth>=2.15.0->google-generativeai->pgmpy) (5.5.0)\n",
            "Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth>=2.15.0->google-generativeai->pgmpy) (0.4.1)\n",
            "Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth>=2.15.0->google-generativeai->pgmpy) (4.9)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from patsy>=0.5.6->statsmodels->pgmpy) (1.16.0)\n",
            "Requirement already satisfied: httplib2<1.dev0,>=0.19.0 in /usr/local/lib/python3.10/dist-packages (from google-api-python-client->google-generativeai->pgmpy) (0.22.0)\n",
            "Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in /usr/local/lib/python3.10/dist-packages (from google-api-python-client->google-generativeai->pgmpy) (0.2.0)\n",
            "Requirement already satisfied: uritemplate<5,>=3.0.1 in /usr/local/lib/python3.10/dist-packages (from google-api-python-client->google-generativeai->pgmpy) (4.1.1)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch->pgmpy) (2.1.5)\n",
            "Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.10/dist-packages (from pydantic->google-generativeai->pgmpy) (0.7.0)\n",
            "Requirement already satisfied: pydantic-core==2.23.4 in /usr/local/lib/python3.10/dist-packages (from pydantic->google-generativeai->pgmpy) (2.23.4)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->torch->pgmpy) (1.3.0)\n",
            "Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /usr/local/lib/python3.10/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.6->google-generativeai->pgmpy) (1.64.1)\n",
            "Requirement already satisfied: grpcio-status<2.0.dev0,>=1.33.2 in /usr/local/lib/python3.10/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.6->google-generativeai->pgmpy) (1.48.2)\n",
            "Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai->pgmpy) (0.6.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (2.2.3)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (2024.8.30)\n"
          ]
        }
      ],
      "source": [
        "!pip install pgmpy\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "**2. Import Libraries:**"
      ],
      "metadata": {
        "id": "hO9M_r1u27HL"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "from pgmpy.models import BayesianNetwork\n",
        "from pgmpy.factors.discrete import TabularCPD\n",
        "from pgmpy.inference import VariableElimination\n",
        "from pgmpy.inference import BeliefPropagation\n"
      ],
      "metadata": {
        "id": "qPoBVc7g2-0L"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Exercise 2: Building a Simple Bayesian Network**"
      ],
      "metadata": {
        "id": "W0BW8pZk3fLK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1. Define the Structure:**"
      ],
      "metadata": {
        "id": "FJfCk5cy3jLy"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Define the Bayesian Network structure\n",
        "model = BayesianNetwork([('Weather', 'Traffic'), ('Traffic', 'Late')])\n"
      ],
      "metadata": {
        "id": "fBDVVaJ43eD6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**2. Define Conditional Probability Tables (CPTs):**"
      ],
      "metadata": {
        "id": "QaQHo3fK3rDb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# CPD for Weather: P(Weather)\n",
        "cpd_weather = TabularCPD(variable='Weather', variable_card=2, values=[[0.7], [0.3]],\n",
        "                         state_names={'Weather': ['Sunny', 'Rainy']})\n",
        "\n",
        "# CPD for Traffic: P(Traffic | Weather)\n",
        "cpd_traffic = TabularCPD(variable='Traffic', variable_card=2,\n",
        "                         values=[[0.8, 0.4], [0.2, 0.6]],\n",
        "                         evidence=['Weather'], evidence_card=[2],\n",
        "                         state_names={'Traffic': ['Light', 'Heavy'], 'Weather': ['Sunny', 'Rainy']})\n",
        "\n",
        "# CPD for Late: P(Late | Traffic)\n",
        "cpd_late = TabularCPD(variable='Late', variable_card=2,\n",
        "                      values=[[0.9, 0.3], [0.1, 0.7]],\n",
        "                      evidence=['Traffic'], evidence_card=[2],\n",
        "                      state_names={'Late': ['On Time', 'Late'], 'Traffic': ['Light', 'Heavy']})\n",
        "\n",
        "# Add CPDs to the model\n",
        "model.add_cpds(cpd_weather, cpd_traffic, cpd_late)\n",
        "\n",
        "# Verify if the model is valid\n",
        "assert model.check_model()\n"
      ],
      "metadata": {
        "id": "tiOKKjP43tKL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Exercise 3: Querying the Bayesian Network**"
      ],
      "metadata": {
        "id": "1l6g0Zyf3pRb"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1. Perform Exact Inference:**"
      ],
      "metadata": {
        "id": "ezrfHIn03656"
      }
    },
    {
      "source": [
        "# Perform inference\n",
        "inference = VariableElimination(model)\n",
        "\n",
        "# Query: P(Late | Weather=Rainy)\n",
        "result = inference.query(variables=['Late'], evidence={'Weather': 'Rainy'}) # Use 'Rainy' instead of 1\n",
        "print(result)"
      ],
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "fLLhhoSj4O4y",
        "outputId": "08cea260-88c5-4351-d336-5db8f23691c0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "+---------------+-------------+\n",
            "| Late          |   phi(Late) |\n",
            "+===============+=============+\n",
            "| Late(On Time) |      0.5400 |\n",
            "+---------------+-------------+\n",
            "| Late(Late)    |      0.4600 |\n",
            "+---------------+-------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Exercise 4: Parameter Learning**\n"
      ],
      "metadata": {
        "id": "xzhMpNkD4SSK"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1. Simulate a Dataset:**"
      ],
      "metadata": {
        "id": "iB-zQU294V7y"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "data = pd.DataFrame(data={'Weather': np.random.choice(['Sunny', 'Rainy'], size=1000, p=[0.7, 0.3]),\n",
        "                          'Traffic': np.random.choice(['Light', 'Heavy'], size=1000, p=[0.6, 0.4]),\n",
        "                          'Late': np.random.choice(['On Time', 'Late'], size=1000, p=[0.8, 0.2])})\n"
      ],
      "metadata": {
        "id": "OxS9RxTH4Yi6"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# Create a synthetic dataset # for Sunny, 1 for Rainy\n",
        "data = pd.DataFrame({\n",
        "    'Weather': np.random.choice([0, 1], size=1000, p=[0.8, 0.2]), 'Traffic': np.nan,\n",
        "    'Late': np.nan\n",
        "})\n",
        "# Fill Traffic based on Weather\n",
        "data.loc[data['Weather'] == 0, 'Traffic'] = np.random.choice(\n",
        "      [0, 1],\n",
        "      size=data[data['Weather']== 0].shape[0],\n",
        "      p=[0.9, 0.1]\n",
        ")\n",
        "data.loc[data['Weather'] == 1, 'Traffic'] = np.random.choice(\n",
        "      [0, 1],\n",
        "      size=data[data['Weather']==1].shape[0],\n",
        "      p=[0.5, 0.5]\n",
        ")\n",
        "# Fill Late based on Traffic\n",
        "data['Late'] = np.where(\n",
        "    data['Traffic'] == 0,\n",
        "    np.random.choice([0, 1], size=data.shape[0], p=[0.95, 0.05]),\n",
        "    np.random.choice([0, 1], size=data.shape[0], p=[0.4, 0.6])\n",
        ")"
      ],
      "metadata": {
        "id": "LqR9bNAz5L4a"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**2. Estimate the Parameters:**\n"
      ],
      "metadata": {
        "id": "FSY_cET053VB"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from pgmpy.estimators import MaximumLikelihoodEstimator\n",
        "\n",
        "# Create a new model with the same structure\n",
        "model = BayesianNetwork([('Weather', 'Traffic'), ('Traffic', 'Late')])\n",
        "\n",
        "# Estimate the CPDs from the data\n",
        "model.fit(data, estimator=MaximumLikelihoodEstimator)\n",
        "\n",
        "# Print learned CPDs\n",
        "for cpd in model.get_cpds():\n",
        "    print(cpd)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "20MG0AQQ54bS",
        "outputId": "c7fa6cf4-c805-49dd-ecb7-76bdc4c71870"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "+------------+-------+\n",
            "| Weather(0) | 0.795 |\n",
            "+------------+-------+\n",
            "| Weather(1) | 0.205 |\n",
            "+------------+-------+\n",
            "+--------------+---------------------+---------------------+\n",
            "| Weather      | Weather(0)          | Weather(1)          |\n",
            "+--------------+---------------------+---------------------+\n",
            "| Traffic(0.0) | 0.8968553459119497  | 0.47804878048780486 |\n",
            "+--------------+---------------------+---------------------+\n",
            "| Traffic(1.0) | 0.10314465408805032 | 0.5219512195121951  |\n",
            "+--------------+---------------------+---------------------+\n",
            "+---------+---------------------+---------------------+\n",
            "| Traffic | Traffic(0.0)        | Traffic(1.0)        |\n",
            "+---------+---------------------+---------------------+\n",
            "| Late(0) | 0.9395807644882861  | 0.41798941798941797 |\n",
            "+---------+---------------------+---------------------+\n",
            "| Late(1) | 0.06041923551171394 | 0.582010582010582   |\n",
            "+---------+---------------------+---------------------+\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##**Exercise 5: Visualizing the Bayesian Networkt**"
      ],
      "metadata": {
        "id": "iKq1e0oL6F0y"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1. Visualize the Network Structure:**\n"
      ],
      "metadata": {
        "id": "LPhe4DAY6IAJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import networkx as nx\n",
        "\n",
        "# Create a NetworkX graph from the Bayesian Network\n",
        "nx_graph = nx.DiGraph()\n",
        "\n",
        "# Add nodes\n",
        "for cpd in model.get_cpds():\n",
        "    nx_graph.add_node(cpd.variable)\n",
        "\n",
        "# Add edges based on the structure of the Bayesian Network\n",
        "for parent, child in model.edges():\n",
        "    nx_graph.add_edge(parent, child)\n",
        "\n",
        "# Draw the graph\n",
        "plt.figure(figsize=(8, 6))\n",
        "pos = nx.spring_layout(nx_graph)  # Define the layout for the nodes\n",
        "nx.draw(\n",
        "    nx_graph, pos,\n",
        "    with_labels=True,\n",
        "    node_color='lightblue',\n",
        "    font_weight='bold',\n",
        "    arrows=True\n",
        ")\n",
        "plt.title('Bayesian Network Structure')\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 659
        },
        "id": "3cAqtOJM7VyR",
        "outputId": "e63a1a30-a5b0-4714-f878-4541c4274264"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 800x600 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAzMAAAKCCAYAAADlSofSAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABgoElEQVR4nO3dd3hUdd7//9ekF0gCJKFK70VApJfQUybursuqiw1FFEEEk3vb9773Z9ly773q7lpWwQp2ERRZZlIgtNClhN4hQVqAAKEkkDbn9wfLWWMCBEhyMpPn47q8lsycOfOaGPfKi/f5fI7NMAxDAAAAAOBmvKwOAAAAAAC3gjIDAAAAwC1RZgAAAAC4JcoMAAAAALdEmQEAAADgligzAAAAANwSZQYAAACAW6LMAAAAAHBLlBkAAAAAbokyAwBVzGaz6cUXX7Q6hscbOnSounbtanUMAEA1oswAqJFmzZolm81W6p/IyEgNGzZMycnJVsdzCy+++KJsNpsaNmyo/Pz8Ms+3bNlS8fHxt3Tut99+W7NmzbrNhDXLggULFBUVpcjISAUFBal169a6//77lZKSYh5z7Ngxvfjii9q8ebNlOWtCBgCoKSgzAGq0P/zhD/rkk0/08ccf6ze/+Y1OnTqluLg4ORwOq6NV2KVLl/T73//esvc/efKkpk+fXqnn9LQy8+qrr+onP/mJbDab/t//+3/6xz/+oTFjxmjfvn368ssvzeOOHTuml156yfIyY3UGAKgpfKwOAADXExsbq7vvvtv8+oknnlDDhg31xRdf3PJUoboFBARY+v49evTQK6+8osmTJyswMNDSLFUhLy9PwcHBt/z64uJi/fGPf9SoUaO0cOHCMs+fPHnyls+dn5+voKCgW359dbp8+bL8/Pzk5cXfcwJwH/w/FgC3EhYWpsDAQPn4lP67mFdffVUDBgxQgwYNFBgYqF69emnu3LmljomKilL37t3LPW+HDh0UHR1tfu1yufTaa6+pS5cuCggIUMOGDTVx4kSdPXu21Os2bNig6OhohYeHKzAwUK1atdL48eNLHfPjNTOHDh3S5MmT1aFDBwUGBqpBgwa67777lJWVVep1Vy+1W7VqlRITExUREaHg4GDde++9OnXqVEW/ZXr++ed14sSJCk1nKvK5W7ZsqR07dmj58uXmJYBDhw5Vbm6uvL299cYbb5jH5uTkyMvLSw0aNJBhGObjkyZNUqNGjUq995w5c9SrVy8FBgYqPDxcDz/8sI4ePVrqmMcee0x16tTRgQMHFBcXp7p16+qhhx665udZuHChgoKCNHbsWBUXF5d7TE5Ojs6fP6+BAweW+3xkZKQkadmyZerdu7ck6fHHHzc/+9UJ1dU1Oxs3btSQIUMUFBSk//7v/5Z07XVTLVu21GOPPVbqsdzcXCUkJKhly5by9/dXs2bN9OijjyonJ+eGGco739VsQ4cONb9etmyZbDabvvzyS/3+979X06ZNFRQUpPPnz0uS1q1bp5iYGIWGhiooKEhRUVFatWpVud8fALASkxkANdq5c+eUk5MjwzB08uRJvfnmm7p48aIefvjhUse9/vrr+slPfqKHHnpIhYWF+vLLL3XffffJ4XDIbrdLkh555BE9+eST2r59e6mF4uvXr9fevXtLXQo2ceJEzZo1S48//rimTp2qzMxM/fOf/1RGRoZWrVolX19fnTx5UqNHj1ZERIR+97vfKSwsTFlZWfrmm2+u+5nWr1+v1atX65e//KWaNWumrKwsTZ8+XUOHDtXOnTvL/E3+s88+q3r16umFF15QVlaWXnvtNU2ZMkWzZ8+u0Pdw8ODBGj58uF5++WVNmjTputOZinzu1157Tc8++6zq1Kmj//mf/5EkNWzYUGFhYeratavS09M1depUSdLKlStls9l05swZ7dy5U126dJEkrVixQoMHDzbf9+p79u7dW3/5y1904sQJvf7661q1apUyMjIUFhZmHltcXKzo6GgNGjRIr7766jUnHw6HQ7/4xS/0wAMP6MMPP5S3t3e5x0VGRiowMFALFizQs88+q/r165d7XKdOnfSHP/xBzz//vJ566ikz/4ABA8xjTp8+rdjYWP3yl7/Uww8/rIYNG17ze12eixcvavDgwdq1a5fGjx+vu+66Szk5OfrXv/6lI0eOVCjDzfjjH/8oPz8//epXv1JBQYH8/Py0ZMkSxcbGqlevXnrhhRfk5eWlmTNnavjw4VqxYoX69OlzS+8FAFXCAIAaaObMmYakMv/4+/sbs2bNKnN8fn5+qa8LCwuNrl27GsOHDzcfy83NNQICAozf/va3pY6dOnWqERwcbFy8eNEwDMNYsWKFIcn47LPPSh2XkpJS6vF58+YZkoz169df97NIMl544YVrZjUMw1izZo0hyfj4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v/++WQaefvrpUs+99tpr5nP9+vUr9VxaWpr5XHx8/E2/b2ZuvnbmXKzQsbPffFWz33xVjo/eu+n3qa24+A4AAAAer3///uaf16xZU+q5H36dkZGhgoIC+fv7l3nux0XnRvIKi7Xl5LkKH//VW3+XJEU0aab4cU9e87jCEtdN5fBkTGYAAADg8Tp16qSQkBBJ0vbt23XhwgXzubVr15p/LiwsVEZGhvn17ZSZjBNX1shUtuyLlyv/pNeQl5dXbe91KygzAAAA8HheXl7q27evJMnlcum7776TJB0/flzff/+9JKlz586S/lNuDMPQunXrzNf36dNHkrR161aNHTtWjRs3lp+fn5o2baoJEyboyJEj5vudLyjShs1b9Y9fPaNp9iiN69tZ93dtrscHdNOfJz6iHev/U6Bmv/mqxnRsYn596tgRjenYRGM6NtHTw/uU+Sx5RSX66PMvdOedd8rf31/t27fXV199Vea4U6dOKTExUe3atZO/v7/q1asnu91eqrxJ0rJly8xL6R577DF988036tGjh/z9/fXKK6/c5He6elFmAAAAUCuUd6nZ1f9t166d7HZ7qcf27t2rM2fOSPrPZCc5OVl9+vTRl19+qezsbBUVFenYsWP64IMP1Lt3b2VmZkq6slbm8L7dWuGYpyMH9uniuVyVFBfr/JnT2rR8sV4c9wttW7vqlj7HquR/6bGHHtS2bdtUWFioffv2aezYsdqzZ495zPfff6+77rpL//jHP7R//34VFhYqNzdXSUlJGjJkiP71r3+Ve+709HT94he/0JYtW1RYWHhL+aoTZQYAAAC1wg8vE7taWK5OKfr166cBAwaUeuzHl5jl5+dr3LhxKigokI+Pj/785z9r4cKF+s1vfiNJys7O1uTJk6/8Oa9AjVu10bjfvqDfvvWhXpw1Ry/O+kpPvfh/8vXzl8vl0rx335QkDR/zS/3ps3nme4VFROpPn83Tnz6bp1+9XnYzgKMH9yv6/gflcDg0YsQISVemTe+//755zOTJk81J0aOPPqqUlBRNnz5dderUUVFRkcaPH1/uJWSZmZm6++67NWfOHH377bcaPHjwTX2PqxsbAAAAAKBW6Nevn2w2mwzD0Nq1a83/la5Mba5Obr7//nsdP368TJlZuHChTp06JUkaNWqUhgwZIkm655579NVXXykrK0upqak6fvKk8opK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\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}