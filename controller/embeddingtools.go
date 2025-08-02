// Copyright (c) 2025 Reza Arani
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package aillm

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/redis/go-redis/v9"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/textsplitter"
)

// SplitText method to split the text into smaller chunks.
//
// This function takes the text stored in the LLMTextEmbedding struct and splits it
// into smaller chunks based on the specified chunk size and overlap settings.
// The resulting chunks are stored as schema.Document objects.
//
// Returns:
//   - []schema.Document: A slice containing the split document chunks.
//   - error: An error if the text splitting process encounters any issues.
func (emb *LLMTextEmbedding) SplitText() ([]schema.Document, error) {
	// Create a new text loader with the provided input text
	p := documentloaders.NewText(strings.NewReader(emb.Text))
	// Initialize a recursive character-based text splitter

	split := textsplitter.NewRecursiveCharacter()
	split.ChunkSize = emb.ChunkSize       // Define the maximum size of each chunk
	split.ChunkOverlap = emb.ChunkOverlap // Define the overlap between chunks
	// Split the text using the specified chunking parameters
	docs, err := p.LoadAndSplit(context.Background(), split)
	// Store the resulting chunks in the EmbeddedDocuments field
	emb.EmbeddedDocuments = docs
	return docs, err
}

const splitPrompt = `You are a helpful assistant that splits long text documents into chunks of approximately %d words. 
Each chunk must:
- Contain complete sentences only (do not break sentences between chunks).
- Be as close as possible to %d words, but sentence integrity is more important.
- End with a list of keywords that summarize the chunk content, written as a comma-separated list.

Format:
Each chunk must begin with this label on its own line: ` + "`----CHUNK----`" + `
Then, include the chunk content.
Then, on a new line, include the keyword list using the format: ` + "`#keywords:`" + ` keyword1, keyword2, ...

Example:

----CHUNK----
[Chunk text here.]

###keywords:### keyword1, keyword2, keyword3

Now, here is the document to split and annotate:
%v
`

func splitTextIntoFixedSizedChunks(rawText string, chunkSize int) []string {
	var chunks []string

	// check if the text is smaller than the chunk size
	if len(rawText) <= chunkSize {
		return []string{rawText}
	}

	start := 0
	for start < len(rawText) {
		// set the end of the chunk, if it is greater than the text length, set it to the end of the text
		end := start + chunkSize
		if end > len(rawText) {
			end = len(rawText)
		}

		// find the nearest point to the end of the chunk
		chunk := rawText[start:end]

		// check if the chunk is not finished with a "."
		if end < len(rawText) && rawText[end] != ' ' {
			// find the last space to prevent the word from being split
			lastSpace := strings.LastIndex(chunk, " ")
			if lastSpace != -1 {
				end = start + lastSpace
				chunk = rawText[start:end]
			}
		}

		// add the chunk to the list
		chunks = append(chunks, strings.TrimSpace(chunk))

		// set the start of the next chunk
		start = end
	}

	return chunks
}

func (emb *LLMTextEmbedding) SplitTextWithLLM() (docs []schema.Document, keywords []string, inconsistentChunks map[int]string, err error) {
	// Split the large text into chunks to avoid token limits (optional)
	chunks := splitTextIntoFixedSizedChunks(emb.Text, emb.ChunkSize)
	resultChunks := []schema.Document{}
	inconsistentChunks = make(map[int]string)

	for _, chunk := range chunks {
		// Use the new prompt with both chunking and keyword extraction
		prompt := fmt.Sprintf(splitPrompt, emb.ChunkSize, emb.ChunkSize, chunk)
		resp, err := emb.lLMContainer.AskLLM("", emb.lLMContainer.WithExactPrompt(prompt), emb.lLMContainer.WithAllowHallucinate(true))
		if err != nil {
			return nil, keywords, inconsistentChunks, err
		}

		chunksArray := strings.Split(resp.Response.Choices[0].Content, "----CHUNK----")

		for idx, chunkItem := range chunksArray {
			chunkItem = strings.TrimSpace(chunkItem)
			if len(strings.Fields(chunkItem)) < 3 {
				continue
			}
			resultChunks = append(resultChunks, schema.Document{PageContent: chunkItem})
			// Validate original content presence (optional)
			content := strings.Split(chunkItem, "###keywords:### ")

			contentOnly := trimContent(content[0])

			if !strings.Contains(emb.Text, strings.TrimSpace(contentOnly)) {
				inconsistentChunks[idx] = chunkItem
			}
			if len(content) > 0 {
				generatedkeywords := strings.Split(trimContent(content[1]), ",")
				for idx, keyword := range generatedkeywords {
					generatedkeywords[idx] = strings.TrimSpace(keyword)
				}
				keywords = append(keywords, generatedkeywords...)
			}
		}
	}

	return resultChunks, keywords, inconsistentChunks, nil
}

func trimContent(content string) string {
	//use regex in future
	content = strings.Trim(content, "\n")
	content = strings.TrimSpace(content)
	return content
}

// sanitizeRedisKey ensures that a string is safe to be used as a Redis key.
//
// Parameters:
//   - input: The original string to sanitize.
//
// Returns:
//   - string: The sanitized key string with special characters replaced.
func (llm LLMEmbeddingObject) sanitizeRedisKey(input string) string {
	// Replace any non-alphanumeric characters with underscores
	re := regexp.MustCompile(`[^a-zA-Z0-9:_-]`)
	sanitized := re.ReplaceAllString(input, "_")

	// Remove duplicate underscores
	sanitized = strings.ReplaceAll(sanitized, "__", "_")

	// حذف زیرخط‌های اضافی در ابتدا و انتها
	sanitized = strings.Trim(sanitized, "_")

	return sanitized
}

// deleteRedisWildCard deletes keys matching a given wildcard pattern in Redis.
//
// Parameters:
//   - redisClient: The Redis client instance.
//   - k: The key pattern to search and delete.
//
// Returns:
//   - int: The number of keys deleted.
//   - error: An error if the deletion fails.
func (llm *LLMContainer) deleteRedisWildCard(redisClient *redis.Client, k string, addWildCard bool) (int, error) {
	var ctx = context.Background()
	// Replace spaces with underscores for key pattern matching
	// k = strings.ReplaceAll(k, " ", "____")
	re := regexp.MustCompile(`[^a-zA-Z0-9:_-]`)
	k = re.ReplaceAllString(k, "_")
	k = strings.ReplaceAll(k, "__", "_")

	if addWildCard {
		k = k + ":*"
	}
	// Retrieve matching keys
	keys, err := redisClient.Keys(ctx, k).Result()
	if err != nil {
		return 0, err
	}
	// Delete the matching keys
	keyCount := len(keys)
	if len(keys) > 0 {
		_, delErr := redisClient.Del(ctx, keys...).Result()
		if delErr != nil {
			return 0, delErr
		}

	}
	// CacheObject.Delete(k)
	return keyCount, nil
}
