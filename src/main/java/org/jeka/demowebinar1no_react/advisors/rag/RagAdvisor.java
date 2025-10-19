package org.jeka.demowebinar1no_react.advisors.rag;

import lombok.Builder;
import lombok.Getter;
import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.api.AdvisorChain;
import org.springframework.ai.chat.client.advisor.api.BaseAdvisor;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.jeka.demowebinar1no_react.advisors.expension.ExpansionQueryAdvisor.ENRICHED_QUESTION;

@Builder
public class RagAdvisor implements BaseAdvisor {

    @Builder.Default
    private static final PromptTemplate template = PromptTemplate.builder().template("""
            Ты — консультант ресторана японской кухни, который помогает клиентам выбрать подходящие роллы и сеты из меню.
                        
            Твоя цель — помочь гостю подобрать блюдо по его запросу, вкусу, диете или предпочтениям.
                        
            Формат ответа:
            1. Говори вежливо, дружелюбно, как официант-консультант.
            2. В ответе упоминай название, краткое описание, цену, вес и количество штук.
            3. Добавляй эмодзи, чтобы визуально различать блюда (например, 🍣, 🥑, 🐉, 🍤).
            4. Если блюд несколько — выведи их списком, красиво оформленным.
            5. В конце можешь задать уточняющий вопрос, чтобы предложить более точную рекомендацию.
            6. Никогда не упоминай ID, JSON или внутренние данные.
            7. Отвечай только по меню ресторана (роллы, сеты, напитки, соусы). Не отвечай на вопросы, не относящиеся к меню.
                        
            Пример:
            Пользователь: "Какие роллы без рыбы?"
            Ответ:
            "Вот несколько вкусных вариантов без рыбы:
            🍀 Вегетарианский ролл — свежий и лёгкий, с авокадо, огурцом и сливочным сыром. 320 ₽ за 8 шт (210 г).
            🐉 Дракон ролл — с креветкой темпура, авокадо и сыром. 480 ₽ за 8 шт (230 г).
            Хотите, чтобы я подобрала что-то острое или наоборот — мягкое по вкусу?"
            
            CONTEXT: {context}
            Question: {question}
            """).build();



    private VectorStore vectorStore;

    @Builder.Default
    private SearchRequest searchRequest = SearchRequest.builder().topK(5).similarityThreshold(0.62).build();

    @Getter
    private final int order;

    public static RagAdvisorBuilder build(VectorStore vectorStore) {
        return new RagAdvisorBuilder().vectorStore(vectorStore);
    }

    @Override
    public ChatClientRequest before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain) {
        String originalUserQuestion = chatClientRequest.prompt().getUserMessage().getText();
        String queryToRag = chatClientRequest.context().getOrDefault(ENRICHED_QUESTION, originalUserQuestion).toString();

        List<Document> documents = vectorStore
                .similaritySearch(SearchRequest.from(searchRequest).query(queryToRag)
                .topK(searchRequest.getTopK()*2)
                .build());

        if (documents == null || documents.isEmpty()) {
            return chatClientRequest.mutate().context("CONTEXT","ТУТ ПУСТО - ни один документ моя собачка не обнаружила").build();
        }

        BM25RerankEngine rerankEngine = BM25RerankEngine.builder().build();

//        documents = rerankEngine.rerank(documents,queryToRag,searchRequest.getTopK());




        String llmContext = documents.stream().map(Document::getText).collect(Collectors.joining(System.lineSeparator()));

        String finalUserPrompt = template.render(
                Map.of("context", llmContext, "question", originalUserQuestion)
        );


        return chatClientRequest.mutate().prompt(chatClientRequest.prompt().augmentUserMessage(finalUserPrompt)).build();
    }

    @Override
    public ChatClientResponse after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain) {
        return chatClientResponse;
    }


}