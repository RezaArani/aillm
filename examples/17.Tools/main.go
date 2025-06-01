package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"

	aillm "github.com/RezaArani/aillm/controller"
	"github.com/tmc/langchaingo/llms"
)

func main() {
	log.Println("Start:")

	llmclient := &aillm.OpenAIController{
		Config: aillm.LLMConfig{
			APIToken: os.Getenv("OPENAIAPITOKEN"),
			Apiurl:   "https://api.openai.com/v1/",
			AiModel:  "gpt-4o",
		},
	}

	//

	// Create an LLM instance with OllamaClient
	llm := aillm.LLMContainer{
		Embedder:       llmclient,
		LLMClient:      llmclient,
		ScoreThreshold: 0.6,
		RedisClient: aillm.RedisClient{
			Host: "localhost:6379",
		},
		AllowHallucinate: true,
	}

	llm.Init()
	// let's embed some data

	// Now we will ask some questions
	// askKLLM(llm, "give me a list of 3 nearby restaurants?")
	// askKLLM(llm, "What is the weather like in Chicago?")
	askKLLM(llm, "give me list of files in d:\\")

}

func GetTools() aillm.AillmTools {
	handlers := make(map[string]func(interface{}) (string, error))
	handlers["getCurrentWeather"] = getCurrentWeather
	handlers["runCommand"] = runCommand
	return aillm.AillmTools{
		Tools:    availableTools,
		Handlers: handlers,
	}

}
func askKLLM(llm aillm.LLMContainer, query string) {
	log.Println("LLM Reply to " + query + ":")

	queryResult, err := llm.AskLLM(
		query,
		llm.WithTools(GetTools()),
		llm.WithStreamingFunc(print),
	)
	response := queryResult.Response
	if err != nil {
		panic(err)
	}
	log.Println("CompletionTokens: ", response.Choices[0].GenerationInfo["CompletionTokens"])
	log.Println("PromptTokens: ", response.Choices[0].GenerationInfo["PromptTokens"])
	log.Println("TotalTokens: ", response.Choices[0].GenerationInfo["TotalTokens"])

}

func print(ctx context.Context, chunk []byte) error {
	fmt.Print(string(chunk))
	return nil
}

func getCurrentWeather(data any) (string, error) {
	return "64 and sunny", nil
}

// availableTools simulates the tools/functions we're making available for
// the model.
var availableTools = []llms.Tool{
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "getCurrentWeather",
			Description: "Get the current weather in a given location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The city and state, e.g. San Francisco, CA",
					},
				},
				"required": []string{"location"},
			},
		},
	},
	{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "runCommand",
			Description: "Execute a system command with arguments and return the output.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"executable": map[string]any{
						"type":        "string",
						"description": "The system command to execute, e.g., dir or ls",
					},
					"args": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "List of arguments to pass to the command, e.g., ['d:\\']",
					},
				},
				"required": []string{"executable", "args"},
			},
		},
	},
}

// Command execution tool
func runCommand(command any) (string, error) {
	var stdout, stderr bytes.Buffer

	cmdMap := command.(map[string]any)

	exe := cmdMap["executable"].(string)

	rawArgs := cmdMap["args"].([]any)
	args := []string{"/C", exe}

	for _, a := range rawArgs {
		argStr, ok := a.(string)
		if !ok {
			return "", fmt.Errorf("argument is not a string: %v", a)
		}
		args = append(args, argStr)
	}

	cmd := exec.Command("cmd.exe", args...) // برای ویندوز

	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		return "", fmt.Errorf("failed: %v - %s", err, stderr.String())
	}

	return stdout.String(), nil
}
