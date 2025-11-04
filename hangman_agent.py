import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import pickle
import re
from tqdm import tqdm

# ==================== GPU SETUP ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== HANGMAN ENVIRONMENT ====================
class HangmanGame:
    def __init__(self, word):
        self.word = word.lower()
        self.word_length = len(word)
        self.guessed_letters = set()
        self.wrong_guesses = 0
        self.max_wrong = 6
        self.repeated_guesses = 0
        
    def guess(self, letter):
        letter = letter.lower()
        
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            return False, False
        
        self.guessed_letters.add(letter)
        
        if letter in self.word:
            return True, self.is_won()
        else:
            self.wrong_guesses += 1
            return False, self.is_lost()
    
    def get_state(self):
        return ''.join([c if c in self.guessed_letters else '_' for c in self.word])
    
    def is_won(self):
        return all(c in self.guessed_letters for c in self.word)
    
    def is_lost(self):
        return self.wrong_guesses >= self.max_wrong
    
    def is_done(self):
        return self.is_won() or self.is_lost()
    
    def get_available_letters(self):
        return set('abcdefghijklmnopqrstuvwxyz') - self.guessed_letters

# ==================== ENHANCED TRIGRAM HMM ====================
class TrigramHMM:
    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.word_patterns = defaultdict(lambda: defaultdict(int))
        self.total_chars = 0
        self.pattern_cache = {}
        
    def train(self, words):
        print("Training Enhanced Trigram HMM...")
        for word in tqdm(words):
            word = word.lower()
            padded = f"^^{word}$$"
            
            # Store word patterns by length
            pattern = self._get_pattern(word)
            self.word_patterns[len(word)][pattern] += 1
            
            for i, char in enumerate(word):
                if char.isalpha():
                    self.unigram_counts[char] += 1
                    self.total_chars += 1
            
            for i in range(len(padded) - 1):
                if padded[i+1].isalpha():
                    self.bigram_counts[padded[i]][padded[i+1]] += 1
            
            for i in range(len(padded) - 2):
                if padded[i+2].isalpha():
                    self.trigram_counts[padded[i]][padded[i+1]][padded[i+2]] += 1
    
    def _get_pattern(self, word):
        """Convert word to consonant/vowel pattern"""
        vowels = set('aeiou')
        return ''.join(['V' if c in vowels else 'C' for c in word])
    
    def get_letter_probs(self, current_state, available_letters):
        cache_key = (current_state, frozenset(available_letters))
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        probs = defaultdict(float)
        state = f"^{current_state}$"
        weight_sum = 0
        
        # Trigram probabilities with higher weight
        for i in range(len(state) - 2):
            if state[i+1] == '_':
                context = (state[i], state[i+2] if i+2 < len(state) else '$')
                for letter in available_letters:
                    count = self.trigram_counts[context[0]]['_'].get(letter, 0)
                    if context[1] != '_' and context[1] != '$':
                        count += self.trigram_counts[context[0]][letter].get(context[1], 0)
                    
                    total = max(sum(self.trigram_counts[context[0]]['_'].values()), 1)
                    probs[letter] += (count / total) * 3.0
                    weight_sum += 3.0
        
        # Bigram probabilities
        for i in range(len(state) - 1):
            if state[i+1] == '_':
                for letter in available_letters:
                    count = self.bigram_counts[state[i]].get(letter, 0)
                    total = max(sum(self.bigram_counts[state[i]].values()), 1)
                    probs[letter] += (count / total) * 2.0
                    weight_sum += 2.0
        
        # Position-specific unigram
        for letter in available_letters:
            if self.total_chars > 0:
                probs[letter] += (self.unigram_counts[letter] / self.total_chars) * 1.0
                weight_sum += 1.0
        
        # Normalize
        if weight_sum > 0:
            probs = {k: v / weight_sum for k, v in probs.items()}
        
        if not probs or sum(probs.values()) == 0:
            uniform = 1.0 / len(available_letters)
            probs = {letter: uniform for letter in available_letters}
        
        self.pattern_cache[cache_key] = probs
        return probs

# ==================== ENHANCED FREQUENCY MODEL ====================
class FrequencyModel:
    def __init__(self):
        self.letter_freq = defaultdict(int)
        self.position_freq = defaultdict(lambda: defaultdict(int))
        self.length_letter_freq = defaultdict(lambda: defaultdict(int))
        self.vowel_consonant_patterns = defaultdict(int)
        self.total_letters = 0
        
    def train(self, words):
        print("Training Enhanced Frequency Model...")
        for word in tqdm(words):
            word = word.lower()
            for i, char in enumerate(word):
                if char.isalpha():
                    self.letter_freq[char] += 1
                    self.position_freq[i % len(word)][char] += 1
                    self.length_letter_freq[len(word)][char] += 1
                    self.total_letters += 1
    
    def get_letter_probs(self, current_state, word_length, available_letters):
        probs = {}
        state_revealed = [c for c in current_state if c != '_']
        
        for letter in available_letters:
            # Global frequency
            global_freq = self.letter_freq.get(letter, 1) / max(self.total_letters, 1)
            
            # Length-specific frequency
            length_freq = self.length_letter_freq[word_length].get(letter, 1) / max(sum(self.length_letter_freq[word_length].values()), 1)
            
            # Boost vowels if few revealed
            vowels = set('aeiou')
            revealed_vowels = sum(1 for c in state_revealed if c in vowels)
            revealed_consonants = len(state_revealed) - revealed_vowels
            
            boost = 1.0
            if letter in vowels and revealed_vowels < 2:
                boost = 1.5
            elif letter not in vowels and revealed_consonants < 3:
                boost = 1.3
            
            probs[letter] = (global_freq * 0.4 + length_freq * 0.6) * boost
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        return probs

# ==================== SIMPLIFIED DQN ====================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ==================== IMPROVED HYBRID AGENT ====================
class HybridHangmanAgent:
    def __init__(self):
        self.hmm = TrigramHMM()
        self.freq_model = FrequencyModel()
        
        self.state_dim = 35
        self.action_dim = 26
        
        self.policy_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.replay_buffer = ReplayBuffer(30000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997
        self.batch_size = 64
        self.target_update_freq = 1000
        self.steps = 0
        
        # Adjusted weights - favor statistical models more
        self.hmm_weight = 0.50
        self.freq_weight = 0.35
        self.rl_weight = 0.15
        
    def train_models(self, corpus_file):
        with open(corpus_file, 'r') as f:
            words = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(words)} words from corpus")
        self.hmm.train(words)
        self.freq_model.train(words)
    
    def state_to_vector(self, game):
        state_str = game.get_state()
        
        # Core features
        features = [
            game.word_length / 20.0,
            game.wrong_guesses / 6.0,
            len(game.guessed_letters) / 26.0,
            state_str.count('_') / len(state_str),
            (6 - game.wrong_guesses) / 6.0,
        ]
        
        # Letter availability (26 features)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features.append(1.0 if letter not in game.guessed_letters else 0.0)
        
        # Pattern features
        revealed = [c for c in state_str if c != '_']
        vowels = set('aeiou')
        features.extend([
            sum(1 for c in revealed if c in vowels) / max(len(revealed), 1),
            sum(1 for c in revealed if c not in vowels) / max(len(revealed), 1),
            len(set(revealed)) / 26.0,
            1.0 if state_str[0] != '_' else 0.0,
        ])
        
        return torch.FloatTensor(features).to(device)
    
    def get_action(self, game, training=True):
        available_letters = game.get_available_letters()
        if not available_letters:
            return None
        
        # During early training or evaluation, rely more on statistical models
        use_statistical = (training and self.epsilon > 0.5) or not training
        
        if training and random.random() < self.epsilon:
            # Intelligent exploration
            hmm_probs = self.hmm.get_letter_probs(game.get_state(), available_letters)
            freq_probs = self.freq_model.get_letter_probs(game.get_state(), game.word_length, available_letters)
            
            combined = {}
            for letter in available_letters:
                combined[letter] = hmm_probs.get(letter, 0) * 0.6 + freq_probs.get(letter, 0) * 0.4
            
            probs = np.array([combined[letter] for letter in sorted(available_letters)])
            probs = probs / probs.sum()
            return np.random.choice(sorted(available_letters), p=probs)
        
        # Get predictions from all models
        hmm_probs = self.hmm.get_letter_probs(game.get_state(), available_letters)
        freq_probs = self.freq_model.get_letter_probs(game.get_state(), game.word_length, available_letters)
        
        # Get RL Q-values
        state_vector = self.state_to_vector(game)
        with torch.no_grad():
            q_values = self.policy_net(state_vector.unsqueeze(0)).squeeze(0).cpu().numpy()
        
        # Combine scores
        scores = {}
        for letter in available_letters:
            idx = ord(letter) - ord('a')
            hmm_score = hmm_probs.get(letter, 0.0)
            freq_score = freq_probs.get(letter, 0.0)
            
            # Normalize RL score
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                rl_score_norm = (q_values[idx] - q_min) / (q_max - q_min)
            else:
                rl_score_norm = 0.5
            
            # Weighted combination
            combined_score = (self.hmm_weight * hmm_score + 
                            self.freq_weight * freq_score + 
                            self.rl_weight * rl_score_norm)
            scores[letter] = combined_score
        
        best_letter = max(scores.keys(), key=lambda x: scores[x])
        return best_letter
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) < self.batch_size * 2:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        actions = torch.LongTensor([ord(a) - ord('a') for a in actions]).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones).to(device)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path='agent.pkl'):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path='agent.pkl'):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Agent loaded from {path}")

# ==================== IMPROVED TRAINING ====================
def train(agent, corpus_file, num_episodes=60000, early_stop_threshold=0.75):
    with open(corpus_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"\nStarting training for {num_episodes} episodes...")
    best_win_rate = 0
    no_improvement_count = 0
    patience = 8000
    
    episode_rewards = []
    win_rates = []
    
    for episode in tqdm(range(num_episodes)):
        word = random.choice(words)
        game = HangmanGame(word)
        
        total_reward = 0
        state = agent.state_to_vector(game)
        
        while not game.is_done():
            action = agent.get_action(game, training=True)
            if action is None:
                break
            
            old_wrong = game.wrong_guesses
            old_repeated = game.repeated_guesses
            
            correct, done = game.guess(action)
            
            # Improved reward shaping
            if game.repeated_guesses > old_repeated:
                reward = -5.0
            elif correct:
                # Reward based on remaining lives
                reward = 3.0 + (6 - game.wrong_guesses) * 0.5
            else:
                reward = -2.0
            
            if game.is_won():
                reward += 50.0
            elif game.is_lost():
                reward -= 20.0
            
            total_reward += reward
            next_state = agent.state_to_vector(game)
            
            loss = agent.update(state, action, reward, next_state, game.is_done())
            state = next_state
        
        episode_rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Evaluate every 1000 episodes
        if (episode + 1) % 1000 == 0:
            win_rate = evaluate(agent, words, num_games=300)
            win_rates.append(win_rate)
            avg_reward = np.mean(episode_rewards[-1000:])
            
            print(f"\nEpisode {episode+1}: Win Rate = {win_rate:.3f}, Avg Reward = {avg_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                no_improvement_count = 0
                agent.save('best_agent.pkl')
                print(f"✓ New best model saved!")
            else:
                no_improvement_count += 1000
            
            if win_rate >= early_stop_threshold:
                print(f"\n✓ Reached {early_stop_threshold*100}% win rate! Early stopping...")
                break
            
            if no_improvement_count >= patience:
                print(f"\n✓ No improvement for {patience} episodes. Early stopping...")
                break
    
    print(f"\nTraining complete! Best win rate: {best_win_rate:.3f}")
    return agent

def evaluate(agent, words, num_games=300):
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for _ in range(num_games):
        word = random.choice(words)
        game = HangmanGame(word)
        
        while not game.is_done():
            action = agent.get_action(game, training=False)
            if action is None:
                break
            game.guess(action)
        
        if game.is_won():
            wins += 1
        total_wrong += game.wrong_guesses
        total_repeated += game.repeated_guesses
    
    avg_wrong = total_wrong / num_games
    avg_repeated = total_repeated / num_games
    print(f"  Avg Wrong: {avg_wrong:.2f}, Avg Repeated: {avg_repeated:.2f}")
    
    return wins / num_games

# ==================== TESTING ====================
def test(agent, test_file, num_games=2000):
    with open(test_file, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"\nTesting on {num_games} games...")
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for i in tqdm(range(num_games)):
        word = words[i % len(words)]
        game = HangmanGame(word)
        
        while not game.is_done():
            action = agent.get_action(game, training=False)
            if action is None:
                break
            game.guess(action)
        
        if game.is_won():
            wins += 1
        
        total_wrong += game.wrong_guesses
        total_repeated += game.repeated_guesses
    
    success_rate = wins / num_games
    final_score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Games Played: {num_games}")
    print(f"Wins: {wins}")
    print(f"Success Rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"Total Wrong Guesses: {total_wrong}")
    print(f"Total Repeated Guesses: {total_repeated}")
    print(f"Average Wrong per Game: {total_wrong/num_games:.2f}")
    print(f"Average Repeated per Game: {total_repeated/num_games:.2f}")
    print("="*60)
    print(f"FINAL SCORE: {final_score:.2f}")
    print("="*60)
    
    return final_score

# ==================== MAIN ====================
def main():
    print("="*60)
    print("IMPROVED HYBRID HANGMAN AGENT")
    print("="*60)
    
    agent = HybridHangmanAgent()
    
    print("\nPhase 1: Training HMM and Frequency Models...")
    agent.train_models('corpus.txt')
    
    print("\nPhase 2: Training RL Agent...")
    agent = train(agent, 'corpus.txt', num_episodes=60000, early_stop_threshold=0.75)
    
    agent.save('final_agent.pkl')
    
    print("\nPhase 3: Testing...")
    final_score = test(agent, 'test.txt', num_games=2000)
    
    print(f"\n🎯 Training complete! Final score: {final_score:.2f}")

if __name__ == "__main__":
    main()