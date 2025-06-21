package aillm

import (
	"context"
	"strings"

	"github.com/tmc/langchaingo/llms"
)

func (llm *LLMContainer) IsQuerySafe(Query string) (bool, TokenUsage, error) {
	llmclient, err := llm.LLMClient.NewLLMClient()
	tokenReport := TokenUsage{}
	if err != nil {
		return true, tokenReport, err
	}

	securityResponse, securityErr := llmclient.GenerateContent(context.TODO(),
		[]llms.MessageContent{

			llms.TextParts(llms.ChatMessageTypeHuman,
				strings.Replace(standAloneSecurityCheckPrompt, "{{User query}}", Query, 1),
			),
		},
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			tokenReport.OutputTokens++
			return nil
		}),
		llms.WithTemperature(0))
	if securityErr != nil {
		return true, tokenReport, securityErr
	}

	isSecure := securityResponse.Choices[0].Content == "1"

	return isSecure, tokenReport, nil
}
