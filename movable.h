#pragma once

// not allowed to clone nor move
class Immobile
{
protected:
    Immobile() = default;
    virtual ~Immobile() = default;

    Immobile(Immobile const &) = delete;
    Immobile(Immobile &&) = delete;
    Immobile& operator=(Immobile const &x) = delete;
    Immobile& operator=(Immobile &&x) = delete;
};

// allowed to move, but not clone
class NoClone
{
protected:
    NoClone() = default;
    virtual ~NoClone() = default;
    NoClone(NoClone &&) = default;
    NoClone& operator=(NoClone &&x) = default;

    NoClone(NoClone const &) = delete;
    NoClone& operator=(NoClone const &x) = delete;
};

