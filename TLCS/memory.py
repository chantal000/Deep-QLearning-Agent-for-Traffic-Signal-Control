import random

class Memory:
    def __init__(self, size_max, size_min):
        self._size_max = size_max
        self._size_min = size_min
        


class NormalMemory(Memory):
    def __init__(self, size_max, size_min):
        self._samples = []
        super().__init__(size_max, size_min)
    
    
    def add_sample(self, sample):
        """
        Add a sample into the memory
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element


    def get_samples(self, n):
        """
        Get n samples randomly from the memory
        """
        if self._size_now() < self._size_min:
            return []

        if n > self._size_now():
            return random.sample(self._samples, self._size_now())  # get all the samples
        else:
            return random.choices(self._samples, k=n)  # get "batch size" number of samples


    def _size_now(self):
        """
        Check how full the memory is
        """
        return len(self._samples)






class SequenceMemory(Memory):
    def __init__(self, size_max, size_min, sequence_length):
        self._sequence_length = sequence_length
        self._buffer = []                       # will store sequences of samples while the episode is not yet done
        self._samples = []                    #we will store each episode sequence in a different array
        super().__init__(size_max, size_min)
        
    
    
    def add_to_buffer(self, sample):
        """
        Samples will first be added to a buffer, as long as the episode is not finished yet.
        After the episode is done, the buffer will be added to the memory.
        """
        self._buffer.append(sample)
     
     
    def _collect_and_empty_buffer(self):
        temp = self._buffer
        self._buffer = []
        
        return temp
    
    
    
    
    def add_sequence(self):
        """
        Add the finished episode sequence from the buffer into the memory
        """
        sequence = self._collect_and_empty_buffer()
        self._samples.append(sequence)
        
        
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # if the length is greater than the size of memory, remove the oldest element


    def get_samples(self, batch_size): #TO DO
        """
        Get batch_size samples randomly from the memory
        """
        if self._size_now() < self._size_min:
            return []

        sampled_episodes = random.choices(self._samples, k=batch_size)
        sampled_traces = []        
        
        for episode in sampled_episodes:
            start_point = random.randint(0,len(episode)-self._sequence_length)
            sampled_traces.append(episode[start_point:start_point+self._sequence_length])
        
        return sampled_traces
        
            


    
    def _size_now(self):
        """
        Check how full the memory is
        """
        return sum(len(x) for x in self._samples)    
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    