(define (domain strips-pacman1D)
  (:requirements :strips)

  (:predicates
    (board ?x) (pacman ?x) (ghost ?x) (food ?x) (capsule ?x) 
    (used ?x) (clear ?x)
    (near ?x ?y ) (east ?x ?y))

  (:action put
    :parameters (?b ?x)
    :precondition (and (board ?b) (not (board ?x)) (clear ?b) (not (used(?x))))
    :effect (and (not (clear ?b)) (used ?x)))
)