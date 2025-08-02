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
	"sync"
	"time"
)

// Memory structure to store user memory session data.
//
// This struct keeps track of a user's questions and the session start time.
//
// Fields:
//   - Questions: A slice of strings representing the list of user queries in the session.
//   - MemoryStartTime: A timestamp indicating when the session started.
type Memory struct {
	Questions       []MemoryData // Stores user queries during the session
	MemoryStartTime time.Time    // Timestamp when the session started
	Summary         string       // Summary of the session
}

// Memory structure to store user memory question data.
//
// This struct keeps track of a user's questions and the session data in Redis.
//
// Fields:
//   - Questions: A string representing the user query.
//   - Answer: A string representing the LLM response to the query.
//   - Keys: A slice of strings that keeps keys of Redis vector data related to this question.
type MemoryData struct {
	Question string
	Answer   string
	Keys     []string
	Summary  string
}

// MemoryManager manages session memories with a time-to-live (TTL) mechanism.
//
// This struct is responsible for storing user sessions, providing thread-safe access
// to session data, and cleaning up expired sessions periodically.
//
// Fields:
//   - memoryMap: A map storing session ID as the key and Memory struct as the value.
//   - mu: A mutex to ensure thread-safe operations on the memory map.
//   - ttl: The time-to-live (TTL) duration after which sessions will be removed automatically.
type MemoryManager struct {
	memoryMap map[string]Memory // Stores session data with session ID as the key
	mu        sync.Mutex        // Mutex to prevent concurrent access issues
	ttl       time.Duration     // Session expiration time duration
}

// NewMemoryManager creates and initializes a new MemoryManager with a specified TTL (Time-To-Live).
//
// This function initializes the session memory map and sets up a background process
// to periodically remove expired sessions.
//
// Parameters:
//   - ttlMinutes: The time-to-live duration in minutes before session data expires.
//
// Returns:
//   - *MemoryManager: A pointer to the newly created MemoryManager instance.
func NewMemoryManager(ttlMinutes int) *MemoryManager {
	m := &MemoryManager{
		memoryMap: make(map[string]Memory),                 // Initialize memory map
		ttl:       time.Duration(ttlMinutes) * time.Minute, // Set TTL duration
	}
	go m.cleanupExpiredSessions() // Start the cleanup routine in the background
	return m
}

// AddMemory adds or updates a session's memory in the memory map.
//
// This function stores user queries within a session and ensures thread-safe access
// using a mutex lock to avoid concurrent read/write issues.
//
// Parameters:
//   - sessionID: The unique identifier for the user's session.
//   - questions: A slice of strings containing user queries.
func (m *MemoryManager) AddMemory(sessionID string, questions []MemoryData) {
	m.mu.Lock() // Lock to ensure thread-safe operation
	defer m.mu.Unlock()
	m.memoryMap[sessionID] = Memory{
		Questions:       questions,  // Store the list of user queries
		MemoryStartTime: time.Now(), // Record the session start time
	}
}

// GetMemory retrieves stored session memory for a given session ID.
//
// The function safely reads from the session map and returns the stored memory if it exists.
//
// Parameters:
//   - sessionID: The unique identifier for the user's session.
//
// Returns:
//   - Memory: The stored session data containing questions and timestamp.
//   - bool: A boolean indicating whether the session ID exists in the memory map.
func (m *MemoryManager) GetMemory(sessionID string) (Memory, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	mem, exists := m.memoryMap[sessionID] // Retrieve session memory if exists
	return mem, exists
}

// DeleteMemory removes a user's session memory from the memory map.
//
// This function ensures safe deletion using a mutex lock to prevent data races.
//
// Parameters:
//   - sessionID: The unique identifier for the session to be deleted.
func (m *MemoryManager) DeleteMemory(sessionID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.memoryMap, sessionID)
}

// cleanupExpiredSessions periodically removes expired sessions from the memory map.
//
// This function runs in a background goroutine and executes every 10 minutes to check for expired sessions.
// If a session's elapsed time exceeds the TTL (Time-To-Live) limit, it is removed from the memory map.
func (m *MemoryManager) cleanupExpiredSessions() {
	// Create a ticker that triggers cleanup every 10 minutes

	ticker := time.NewTicker(10 * time.Minute) // Run cleanup every 10 minutes
	defer ticker.Stop()
	for range ticker.C {
		m.mu.Lock()
		for sessionID, mem := range m.memoryMap {
			// Check if the session has expired based on the TTL
			if time.Since(mem.MemoryStartTime) > m.ttl {
				delete(m.memoryMap, sessionID) // Remove expired session
			}
		}
		m.mu.Unlock()
	}
}
