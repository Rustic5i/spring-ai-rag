package org.jeka.demowebinar1no_react.services;

import lombok.Builder;
import org.jeka.demowebinar1no_react.model.Chat;
import org.jeka.demowebinar1no_react.model.ChatEntry;
import org.jeka.demowebinar1no_react.repo.ChatRepository;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.Message;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.Comparator;
import java.util.List;

@Builder
public class PostgresChatMemory implements ChatMemory {

    private ChatRepository chatMemoryRepository;

    private int maxMessages;

    @Override
    public void add(String conversationId, List<Message> messages) {
        Chat chat = chatMemoryRepository.findById(Long.valueOf(conversationId)).orElseThrow();
        for (Message message : messages) {
            chat.addChatEntry(ChatEntry.toChatEntry(message));
        }
        chatMemoryRepository.save(chat);

    }


    @Override
    public List<Message> get(String conversationId) {
        Chat chat = chatMemoryRepository.findById(Long.valueOf(conversationId)).orElseThrow();
        int messageToSkip = Math.max(0, chat.getHistory().size() - maxMessages);
        return chat.getHistory().stream()
                .skip(messageToSkip)
                .map(ChatEntry::toMessage)
//                .limit(maxMessages)
                .toList();

    }

    @Override
    public void clear(String conversationId) {
        //not implemented
    }
}
