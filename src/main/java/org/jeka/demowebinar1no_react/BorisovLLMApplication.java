package org.jeka.demowebinar1no_react;

import org.jeka.demowebinar1no_react.advisors.expension.ExpansionQueryAdvisor;
import org.jeka.demowebinar1no_react.advisors.rag.RagAdvisor;
import org.jeka.demowebinar1no_react.repo.ChatRepository;
import org.jeka.demowebinar1no_react.services.PostgresChatMemory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.ollama.api.OllamaOptions;
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

    @Autowired
    private ChatModel chatModel;


    @Bean
    public ChatClient chatClient(ChatClient.Builder builder) {
        return builder.defaultAdvisors(
                        ExpansionQueryAdvisor.builder(chatModel).order(0).build()
                        , getHistoryAdvisor(1)
                        , SimpleLoggerAdvisor.builder().order(2).build()
//                        , getRagAdviser(3)
                        , RagAdvisor.build(vectorStore).order(3).build()
                        , SimpleLoggerAdvisor.builder().order(4).build()
                )
                .defaultOptions(OllamaOptions.builder()
                        .temperature(0.3)
                        .topP(0.7)
                        .topK(20)
                        .repeatPenalty(1.1)
                        .build())
                .build();
    }

    private Advisor getRagAdviser(int order) {
        return QuestionAnswerAdvisor.builder(vectorStore).promptTemplate(MY_PROMPT_TEMPLATE).searchRequest(
                        SearchRequest.builder()
                                .topK(4) //topK(4) означает максимум 4 чанка бери из бд RAGа
                                .similarityThreshold(0.65) //similarityThreshold(0.9) говорит если на 90% чанк похож на вопрос то берем, если ниже то он мне не нужен, то есть тут фильтруем по качеству
                                .build()
                )
                .order(order)
                .build();
    }


    private Advisor getHistoryAdvisor(int order) {
        return MessageChatMemoryAdvisor.builder(getChatMemory()).order(order).build();
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
