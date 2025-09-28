package org.jeka.demowebinar1no_react;

import org.jeka.demowebinar1no_react.repo.ChatRepository;
import org.jeka.demowebinar1no_react.services.PostgresChatMemory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class BorisovLLMApplication {

    private static final PromptTemplate MY_PROMPT_TEMPLATE = new PromptTemplate(
            "{query}\n\n" +
                    "Контекст:\n" +
                    "---------------------\n" +
                    "{question_answer_context}\n" +
                    "---------------------\n\n" +
                    "Отвечай только на основе контекста выше. Если информации нет в контексте, сообщи, что не можешь ответить."
    );



    @Autowired
    private ChatRepository chatRepository;

    @Autowired
    private VectorStore vectorStore;


    @Bean
    public ChatClient chatClient(ChatClient.Builder builder) {
        return builder.defaultAdvisors(getHistoryAdvisor()
                , getRagAdviser()
        ).build();
    }

    private Advisor getRagAdviser() {
        return QuestionAnswerAdvisor.builder(vectorStore).promptTemplate(MY_PROMPT_TEMPLATE).searchRequest(
                SearchRequest.builder().topK(4).build()
        ).build();
    }


    private Advisor getHistoryAdvisor() {
       return MessageChatMemoryAdvisor.builder(getChatMemory()).order(-10).build();
    }

    private ChatMemory getChatMemory() {
       return PostgresChatMemory.builder()
                .maxMessages(12)
                .chatMemoryRepository(chatRepository)
                .build();
    }


    public static void main(String[] args) {
        ConfigurableApplicationContext context = SpringApplication.run(BorisovLLMApplication.class, args);
        ChatClient chatClient = context.getBean(ChatClient.class);
//        System.out.println(chatClient.prompt().user("Дай первую строчку Bohemian Rhapsody").call().content());
    }



}
