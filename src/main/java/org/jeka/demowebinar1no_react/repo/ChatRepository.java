package org.jeka.demowebinar1no_react.repo;

import org.jeka.demowebinar1no_react.model.Chat;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ChatRepository extends JpaRepository<Chat, Long> {

}
