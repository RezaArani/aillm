// Copyright (c) 2025 John Doe
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
func (llm *LLMContainer) deleteRedisWildCard(redisClient *redis.Client, k string) (int, error) {
	var ctx = context.Background()
	// Replace spaces with underscores for key pattern matching
	// k = strings.ReplaceAll(k, " ", "____")
	re := regexp.MustCompile(`[^a-zA-Z0-9:_-]`)
	k = re.ReplaceAllString(k, "_")
	k = strings.ReplaceAll(k, "__", "_")

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
