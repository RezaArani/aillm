package aillm

import (
	"context"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

func (llm *LLMContainer) IsQuerySafe(Query string, debug bool) (bool, TokenUsage, string, error) {
	llmclient, err := llm.LLMClient.NewLLMClient()
	warning := ""
	tokenReport := TokenUsage{}
	if err != nil {
		return true, tokenReport, warning, err
	}
	prompt := standAloneSecurityCheckPrompt
	if debug {
		prompt = standAloneSecurityCheckPromptForDebugging
	}
	securityResponse, securityErr := llmclient.GenerateContent(context.TODO(),
		[]llms.MessageContent{

			llms.TextParts(llms.ChatMessageTypeHuman,
				strings.Replace(prompt, "{{User query}}", Query, 1),
			),
		},
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			tokenReport.OutputTokens++
			return nil
		}),
		llms.WithTemperature(0.01))
	if securityErr != nil {
		return true, tokenReport, warning, securityErr
	}

	isSecure := strings.HasPrefix(securityResponse.Choices[0].Content, "1")
	if !isSecure && debug {
		warning = securityResponse.Choices[0].Content
	}
	return isSecure, tokenReport, warning, nil
}
